# coding:utf8
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import cPickle as pickle
import math
import heapq
import os
import sys
from gensim.models.word2vec import Word2Vec
import string

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2];

epochs = 25
learn_rate = 0.0001;
batch_size = 256;
dim_hidden = 100;
num_perspective = 1;

num_negs = 1;
neg_sample_size = batch_size * num_negs;
last_layer_size = dim_hidden;

def parse(path):
    g = open(path, 'r')
    for l in g:
        yield eval(l)

def read_data(file_name):

    ratings = [];
    set_users = {};
    set_items = {};
    user_map_item = {};
    namer_user = {};
    namer_item = {};
    rDic = {}
    model = Word2Vec.load('MyModel')
    for l in parse(file_name):
        triplet = []
        triplet.append(l['reviewerID'])
        triplet.append(l['asin'])
        triplet.append(l['overall'])
        triplet.append(l['unixReviewTime'])
        rVec = []
        rText = l['reviewText'].translate(string.maketrans('',''),string.punctuation)
        rText = rText.split(' ')
        for word in rText:
            rVec.append(model[word])
        if(not rDic.has_key(triplet[0])):
            rDic[triplet[0]] = {}
        rDic[triplet[0]][triplet[1]] = rVec
        set_users[triplet[0]] = 1;
        set_items[triplet[1]] = 1;
        if triplet[0] not in namer_user:
            namer_user[triplet[0]] = len(namer_user);
        if triplet[1] not in namer_item:
            namer_item[triplet[1]] = len(namer_item);

        if namer_user[triplet[0]] not in user_map_item:
            user_map_item[namer_user[triplet[0]]] = {};
        user_map_item[namer_user[triplet[0]]][namer_item[triplet[1]]] = float(triplet[2]);

        triplet[0] = namer_user[triplet[0]];
        triplet[1] = namer_item[triplet[1]];
        triplet[2] = float(triplet[2]);
        ratings.append(triplet);

    u_max_num = len(set_users);
    v_max_num = len(set_items);

    flag_user = {};
    latest_item_interaction = {};
    pruned_all_ratings = [];
    for i in xrange(len(ratings)):
        if ratings[len(ratings) - i - 1][0] not in flag_user:
            flag_user[ratings[len(ratings) - i - 1][0]] = 1;
            latest_item_interaction[ratings[len(ratings) - i - 1][0]] = ratings[len(ratings) - i - 1][1];
        else:
            pruned_all_ratings.append(ratings[len(ratings) - i - 1]);
    print ratings[0:10];

    return ratings, rDic, u_max_num, v_max_num, user_map_item, latest_item_interaction, pruned_all_ratings;

# *****************************************************************************************************
file_name = './dataset/' + sys.argv[1];
ratings, rDic, u_max_num, v_max_num, user_map_item, latest_item_interaction, pruned_all_ratings = read_data(file_name)
# *****************************************************************************************************

pruned_user_map_item = {}
pruned_item_map_user = {}
for u, v, r, t in pruned_all_ratings:
    if u not in pruned_user_map_item:
        pruned_user_map_item[u] = {}
    if v not in pruned_item_map_user:
        pruned_item_map_user[v] = {};
    pruned_user_map_item[u][v] = r
    pruned_item_map_user[v][u] = r

for i in xrange(u_max_num):
    if i not in pruned_user_map_item:
        pruned_user_map_item[i] = {};
for i in xrange(v_max_num):
    if i not in pruned_item_map_user:
        pruned_item_map_user[i] = {};

def ADSF_perspective(layer, dim_hidden_in, dim_hidden_out, in_user, in_item):
    user_mat = tf.get_variable(
            "user_mat_" + layer,
            shape = (dim_hidden_in, dim_hidden_out),
            initializer = tf.contrib.layers.xavier_initializer());
    item_mat = tf.get_variable(
            "item_mat_" + layer,
            shape = (dim_hidden_in, dim_hidden_out),
            initializer = tf.contrib.layers.xavier_initializer());
    user_bias = tf.get_variable(
            "user_bias_" + layer,
            shape = [dim_hidden_out],
            initializer = tf.contrib.layers.xavier_initializer());
    item_bias = tf.get_variable(
            "item_bias_" + layer,
            shape = [dim_hidden_out],
            initializer = tf.contrib.layers.xavier_initializer());

    rep_user = tf.nn.relu(tf.matmul(in_user, user_mat) + user_bias);
    rep_item = tf.nn.relu(tf.matmul(in_item, item_mat) + item_bias);

    att_user = tf.get_variable(
            "att_user_" + layer,
            shape = (dim_hidden_out, dim_hidden_out),
            initializer = tf.contrib.layers.xavier_initializer());
    att_item = tf.get_variable(
            "att_item_" + layer,
            shape = (dim_hidden_out, dim_hidden_out),
            initializer = tf.contrib.layers.xavier_initializer());

    rep_user = rep_user * tf.nn.softmax(tf.matmul(rep_item, att_user));
    rep_item = rep_item * tf.nn.softmax(tf.matmul(rep_user, att_item));

    return rep_user, rep_item;

def Perspective(layer, dim_hidden_in, dim_hidden_out, in_user, in_item):
    for i_per in xrange(num_perspective):
        rep_user, rep_item = ADSF_perspective(layer + "_" + str(i_per), dim_hidden_in, dim_hidden_out, in_user, in_item);
        if i_per == 0:
            net_u = rep_user;
            net_v = rep_item;
        else:
            net_u = tf.concat([net_u, rep_user], 1);
            net_v = tf.concat([net_v, rep_item], 1);
    return net_u, net_v;

# DSSM one hots
def inference_neural_DSSM_onehot(one_hot_u, one_hot_v, input_review):
    embed_user = tf.get_variable("embed_user", shape=[u_max_num + 10, dim_hidden], initializer=tf.contrib.layers.xavier_initializer());
    embed_item = tf.get_variable("embed_item", shape=[v_max_num + 10, dim_hidden], initializer=tf.contrib.layers.xavier_initializer());

    rep_u = tf.nn.embedding_lookup(embed_user, one_hot_u);
    rep_v = tf.nn.embedding_lookup(embed_item, one_hot_v);

    lstmCell = tf.contrib.rnn.BasicLSTMCell(64)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, out_keep_prob=0.75)
    value, _ = tf.nn.dynamic_rnn(lstmCell, input_review, dtype=tf.float32)
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    rep = tf.concat([tf.concat(rep_u, last), tf.concat(rep_v, last)], 1);

    w_1 = tf.get_variable(
            "Weight-1",
            shape = (2 * dim_hidden, 2 * dim_hidden),
            initializer = tf.contrib.layers.xavier_initializer());
    b_1 = tf.get_variable(
            "Bias-1",
            shape = [2 * dim_hidden],
            initializer = tf.contrib.layers.xavier_initializer());
    rep = tf.nn.relu(tf.matmul(rep, w_1) + b_1);

    w_4 = tf.get_variable(
            "Weight-4",
            shape = (2 * dim_hidden, 2 * dim_hidden),
            initializer = tf.contrib.layers.xavier_initializer());
    b_4 = tf.get_variable(
            "Bias-4",
            shape = [2 * dim_hidden],
            initializer = tf.contrib.layers.xavier_initializer());
    rep = tf.nn.relu(tf.matmul(rep, w_4) + b_4);

    w = tf.get_variable(
            "Weight-Final",
            shape = (2 * dim_hidden, 1),
            initializer = tf.contrib.layers.xavier_initializer());

    y_dnn = tf.nn.relu(tf.matmul(rep, w));

    y_fm = tf.nn.sigmoid(tf.reduce_sum(rep_u * rep_v, 1, keep_dims=True));

    return tf.nn.sigmoid(y_dnn + y_fm), [];

class MySampler():
    def __init__(self, all_ratings, u_max_num, v_max_num):
        self.sample_con = {}
        self.sample_con_size = 0

        self.all_ratings_map_u = {}
        for u, v, r, t in all_ratings:
            if u not in self.all_ratings_map_u:
                self.all_ratings_map_u[u] = {}
            self.all_ratings_map_u[u][v] = 1

        self.u_max_num = u_max_num
        self.v_max_num = v_max_num

    def smple_one(self):
        u_rand_num = int(np.random.rand() * self.u_max_num)
        v_rand_num = int(np.random.rand() * self.v_max_num)
        if u_rand_num == 0:
            u_rand_num += 1
        if v_rand_num == 0:
            v_rand_num += 1

        if u_rand_num in self.all_ratings_map_u and v_rand_num not in self.all_ratings_map_u[u_rand_num]:
            return u_rand_num, v_rand_num
        else:
            return self.smple_one()


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1.0
    return 0.0


def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0.0


# training
def train_matrix_factorization_With_Feed_Neural():
    top_k = 10
    fen_zhi = tf.reduce_sum(net_u * net_v, 1, keep_dims=True)

    norm_u = tf.sqrt(tf.reduce_sum(tf.square(net_u), 1, keep_dims=True))
    norm_v = tf.sqrt(tf.reduce_sum(tf.square(net_v), 1, keep_dims=True))
    fen_mu = norm_u * norm_v

    return tf.nn.relu(fen_zhi / fen_mu), []


class MySampler():
    def __init__(self, all_ratings, u_max_num, v_max_num):
        self.sample_con = {}
        self.sample_con_size = 0

        self.all_ratings_map_u = {}
        for u, v, r, t in all_ratings:
            if u not in self.all_ratings_map_u:
                self.all_ratings_map_u[u] = {}
            self.all_ratings_map_u[u][v] = 1

        self.u_max_num = u_max_num
        self.v_max_num = v_max_num

    def smple_one(self):
        u_rand_num = int(np.random.rand() * self.u_max_num)
        v_rand_num = int(np.random.rand() * self.v_max_num)
        if u_rand_num == 0:
            u_rand_num += 1
        if v_rand_num == 0:
            v_rand_num += 1

        if u_rand_num in self.all_ratings_map_u and v_rand_num not in self.all_ratings_map_u[u_rand_num]:
            return u_rand_num, v_rand_num
        else:
            return self.smple_one()


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1.0
    return 0.0


def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0.0


# training
def train_matrix_factorization_With_Feed_Neural():
    top_k = 10
    fen_zhi = tf.reduce_sum(net_u * net_v, 1, keep_dims=True)

    norm_u = tf.sqrt(tf.reduce_sum(tf.square(net_u), 1, keep_dims=True))
    norm_v = tf.sqrt(tf.reduce_sum(tf.square(net_v), 1, keep_dims=True))
    fen_mu = norm_u * norm_v

    return tf.nn.relu(fen_zhi / fen_mu), []


class MySampler():
    def __init__(self, all_ratings, u_max_num, v_max_num):
        self.sample_con = {}
        self.sample_con_size = 0

        self.all_ratings_map_u = {}
        for u, v, r, t in all_ratings:
            if u not in self.all_ratings_map_u:
                self.all_ratings_map_u[u] = {}
            self.all_ratings_map_u[u][v] = 1

        self.u_max_num = u_max_num
        self.v_max_num = v_max_num

    def smple_one(self):
        u_rand_num = int(np.random.rand() * self.u_max_num)
        v_rand_num = int(np.random.rand() * self.v_max_num)
        if u_rand_num == 0:
            u_rand_num += 1
        if v_rand_num == 0:
            v_rand_num += 1

        if u_rand_num in self.all_ratings_map_u and v_rand_num not in self.all_ratings_map_u[u_rand_num]:
            return u_rand_num, v_rand_num
        else:
            return self.smple_one()


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1.0
    return 0.0


def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0.0


# training
def train_matrix_factorization_With_Feed_Neural():
    top_k = 10
    final_ndcg_metric_list = []
    final_hr_metric_list = []

    my_sample = MySampler(pruned_all_ratings, u_max_num, v_max_num)

    one_hot_u = tf.placeholder(tf.int32, [None])
    one_hot_v = tf.placeholder(tf.int32, [None])
    true_u_v = tf.placeholder(tf.float32, [None, 1])
    input_review = tf.placeholder(tf.int32, [batch_size, 250])
    pred_val, network_params = inference_neural_DSSM_onehot(one_hot_u, one_hot_v, input_review)

    one_constant = tf.constant(1.0, shape=[1, 1])
    gmf_loss = tf.reduce_mean(-true_u_v * tf.log(pred_val + 1e-10) - (one_constant - true_u_v) * tf.log(one_constant - pred_val + 1e-10))
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(gmf_loss)

    batch_u = np.zeros((batch_size + neg_sample_size)).astype('int32')
    batch_v = np.zeros((batch_size + neg_sample_size)).astype('int32')
    batch_review = np.zeros([batch_size + neg_sample_size, 250, 50], astype('float32'))
    batch_true_u_v = np.zeros((batch_size + neg_sample_size, 1)).astype('float32')

    batch_u_test = np.zeros((100)).astype('int32')
    batch_v_test = np.zeros((100)).astype('int32')
    map_index_u = {}
    map_index_v = {}

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.InteractiveSession(config=config)
    tf.initialize_all_variables().run()

    print "DSSM ONE HOT"
    print "batch      size: ", batch_size
    print "neg sample size: ", neg_sample_size
    print "learn      rate: ", learn_rate
    print "file       name: ", file_name
    print "last layer size: ", last_layer_size
    print "2 layers"
    for epoch in range(epochs):

        np.random.shuffle(pruned_all_ratings)
        one_epoch_loss = 0.0
        one_epoch_batchnum = 0.0
        total_process = len(pruned_all_ratings)/batch_size;
        current_process = 0;
        for index in range(len(pruned_all_ratings) / batch_size):
            train_sample_index = 0

            for i in xrange(batch_size):
                batch_u[i] = pruned_all_ratings[index * batch_size + i][0];
                batch_v[i] = pruned_all_ratings[index * batch_size + i][1];
                batch_review[i] = rDic[batch_u[i]][batch_v[i]]
                batch_true_u_v[i] = 1.0;
            for sam in xrange(neg_sample_size):
                neg_u, neg_v = my_sample.smple_one();
                batch_u[batch_size + i] = neg_u;
                batch_v[batch_size + i] = neg_v;
                batch_review[i] = rDic[batch_u[i]][batch_v[i]]
                batch_true_u_v[batch_size + i] = 0.0;

            _, loss_val, pred_value = sess.run([train_step, gmf_loss, pred_val], feed_dict={one_hot_u: batch_u, one_hot_v: batch_v, input_review: batch_review, true_u_v: batch_true_u_v})

            if index/total_process * 100 > current_process:
                current_process = index/total_process * 100;
                print '*',

            one_epoch_loss += loss_val
            # print batch_true_u_v
            # print loss_val, pred_value
            one_epoch_batchnum += 1.0


            #for j in range(train_sample_index):
            #    u_key = map_index_u[j]
            #    v_key = map_index_v[j]
            #    for v_i in pruned_user_map_item[u_key]:
            #        batch_u[j][v_i] = 0.0
            #    for u_in in pruned_item_map_user.get(v_key, []):
            #        batch_v[j][u_in] = 0.0

            if index == len(pruned_all_ratings) / batch_size -1:
                # print "epoch: ", epoch, " end"
                format_str = '%s: %d epoch, iteration averge loss = %.4f '
                print (format_str % (datetime.now(), epoch, one_epoch_loss / one_epoch_batchnum)),

                # 计算 NDCG@10 与 HR@10
                # evaluate_1
                # evaluate_2
                hr_list = []
                ndcg_list = []
                for u_i in latest_item_interaction:
                    v_latest = latest_item_interaction[u_i]

                    # print u_i, v_latest
                    v_random = [v_latest]
                    i = 1
                    while i < 100:
                        rand_num = int(np.random.rand() * (v_max_num - 1) + 1)
                        if rand_num not in user_map_item[u_i] and rand_num not in v_random and rand_num in pruned_item_map_user:
                            v_random.append(rand_num)
                            i += 1

                    for i in range(100):
                            batch_u_test[i] = u_i;
                            batch_v_test[i] = v_random[i];

                    pred_value = sess.run([pred_val], feed_dict={one_hot_u: batch_u_test, one_hot_v: batch_v_test})
                    pre_real_val = np.array(pred_value).reshape((-1))

                    items = v_random
                    gtItem = items[0]
                    # Get prediction scores
                    map_item_score = {}
                    for i in xrange(len(items)):
                        item = items[i]
                        map_item_score[item] = pre_real_val[i]

                    # Evaluate top rank list
                    # print map_item_score
                    ranklist = heapq.nlargest(top_k, map_item_score, key=map_item_score.get)
                    hr_list.append(getHitRatio(ranklist, gtItem))
                    ndcg_list.append(getNDCG(ranklist, gtItem))

                hr_val, ndcg_val = np.array(hr_list).mean(), np.array(ndcg_list).mean()
                final_hr_metric_list.append(hr_val)
                final_ndcg_metric_list.append(ndcg_val)
                print "RESULT: ", hr_val, ndcg_val

    print 'BEST RESULT: ', max(final_hr_metric_list), max(final_ndcg_metric_list)


if __name__ == "__main__":
    train_matrix_factorization_With_Feed_Neural()
