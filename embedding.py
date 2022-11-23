import tensorflow as tf
import random
import math
import numpy as np
import os
import time
from tensorflow.python.framework import ops
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class _Embedding(object):
    def __init__(self, graph, rep_dim=100, batch_size=1000, negative_ratio=5):
        self.g = graph
        self.rep_dim = rep_dim
        self.node_size = graph.G.number_of_nodes()
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.cur_epoch = 0

        self.gen_sampling_table()

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session()

        self.build_graph()
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def gen_sampling_table(self):
        table_size = 1e8
        power = 0.75
        numNodes = self.node_size

        print("Pre-procesing for non-uniform negative sampling!")
        node_degree = np.zeros(numNodes)  # out degree

        look_up = self.g.look_up_dict
        for edge in self.g.G.edges():
            node_degree[look_up[edge[0]]
                        ] += self.g.G[edge[0]][edge[1]]["weight"]

        norm = sum([math.pow(node_degree[i], power) for i in range(numNodes)])

        self.sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        for j in range(numNodes):
            p += float(math.pow(node_degree[j], power)) / norm
            while i < table_size and float(i) / table_size < p:
                self.sampling_table[i] = j
                i += 1

        data_size = self.g.G.number_of_edges()
        self.edge_alias = np.zeros(data_size, dtype=np.int32)
        self.edge_prob = np.zeros(data_size, dtype=np.float32)
        large_block = np.zeros(data_size, dtype=np.int32)
        small_block = np.zeros(data_size, dtype=np.int32)

        total_sum = sum([self.g.G[edge[0]][edge[1]]["weight"]
                         for edge in self.g.G.edges()])
        norm_prob = [self.g.G[edge[0]][edge[1]]["weight"] *
                     data_size/total_sum for edge in self.g.G.edges()]
        num_small_block = 0
        num_large_block = 0
        for k in range(data_size-1, -1, -1):
            if norm_prob[k] < 1:
                small_block[num_small_block] = k
                num_small_block += 1
            else:
                large_block[num_large_block] = k
                num_large_block += 1
        while num_small_block and num_large_block:
            num_small_block -= 1
            cur_small_block = small_block[num_small_block]
            num_large_block -= 1
            cur_large_block = large_block[num_large_block]
            self.edge_prob[cur_small_block] = norm_prob[cur_small_block]
            self.edge_alias[cur_small_block] = cur_large_block
            norm_prob[cur_large_block] = norm_prob[cur_large_block] + \
                norm_prob[cur_small_block] - 1
            if norm_prob[cur_large_block] < 1:
                small_block[num_small_block] = cur_large_block
                num_small_block += 1
            else:
                large_block[num_large_block] = cur_large_block
                num_large_block += 1

        while num_large_block:
            num_large_block -= 1
            self.edge_prob[large_block[num_large_block]] = 1
        while num_small_block:
            num_small_block -= 1
            self.edge_prob[small_block[num_small_block]] = 1


    def build_graph(self):
        tf.compat.v1.disable_eager_execution()
        self.s = tf.compat.v1.placeholder(tf.int32, [None])
        self.t = tf.compat.v1.placeholder(tf.int32, [None])
        self.sign = tf.compat.v1.placeholder(tf.float32, [None])

        self.emb_vec = tf.Variable(tf.compat.v1.truncated_normal([self.node_size, self.rep_dim], stddev=0.3),
                                name="sn_embedding_vector")
        self.emb_vec = tf.clip_by_norm(self.emb_vec, clip_norm=1, axes=1)

        s_emb = tf.nn.embedding_lookup(self.emb_vec, self.s)
        t_emb = tf.nn.embedding_lookup(self.emb_vec, self.t)

        self.loss = -tf.reduce_mean(tf.compat.v1.log_sigmoid(
            self.sign * tf.reduce_sum(tf.multiply(s_emb, t_emb), axis=1)))

        learning_rate = tf.compat.v1.train.exponential_decay(
            learning_rate=0.001, global_step=self.cur_epoch, decay_steps=20,
            decay_rate=0.9, staircase=True)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

        self.train_op = optimizer.minimize(self.loss)
        print("Initialized")

    def batch_iter(self):
        look_up = self.g.look_up_dict

        table_size = 1e8

        edges = [(look_up[x[0]], look_up[x[1]]) for x in self.g.G.edges()]

        data_size = self.g.G.number_of_edges()
        shuffle_indices = np.random.permutation(np.arange(data_size))

        # positive or negative mod
        mod = 0
        mod_size = 1 + self.negative_ratio
        s = []

        start_index = 0
        end_index = min(start_index+self.batch_size, data_size)
        while start_index < data_size:
            if mod == 0:
                sign = 1.
                s = []
                t = []
                for i in range(start_index, end_index):
                    if not random.random() < self.edge_prob[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    cur_s = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    s.append(cur_s)
                    t.append(cur_t)
            else:
                sign = -1.
                t = []
                for i in range(len(s)):
                    t.append(
                        self.sampling_table[random.randint(0, table_size-1)])

            yield s, t, [sign]
            mod += 1
            mod %= mod_size
            if mod == 0:
                start_index = end_index
                end_index = min(start_index+self.batch_size, data_size)

    def train_one_epoch(self):
        start_time = time.time()
        sum_loss = 0
        batches = self.batch_iter()
        for batch in batches:
            s, t, sign = batch
            feed_dict = {
                self.s: s,
                self.t: t,
                self.sign: sign,
            }
            _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict)
            sum_loss = sum_loss + cur_loss

        end_time = time.time()
        if self.cur_epoch % 100 == 0:
            print('Epoch:{} processing time :{!s} sum of loss:{!s}'.format(
                self.cur_epoch, end_time - start_time, sum_loss))
        self.cur_epoch += 1

    def get_emb(self,name_list):
        embedding = {}
        look_back = self.g.look_back_list
        emb_vec = self.emb_vec.eval(session=self.sess)
        for i, emb in enumerate(emb_vec):
            if look_back[i] in name_list:
                embedding[look_back[i]] = emb
        return embedding


class Embedding(object):
    def __init__(self, graph, namefile1, namefile2,
                 rep_dim=100, batch_size=10000, negative_ratio=5, epoch=500):
        ops.reset_default_graph()
        self.emb_vec = {}
        self.model = _Embedding(graph, rep_dim, batch_size, negative_ratio)
        for i in range(epoch+1):
            self.model.train_one_epoch()
        self.get_emb(namefile1, namefile2)

    def get_emb(self, namefile1, namefile2):
        name_list = []
        f1 = open(namefile1, 'r', encoding='utf-8')
        while 1:
            l = f1.readline().rstrip('\n')
            if l == '':
                break
            name = l+'1'
            name_list.append(name)
        f1.close()
        print('sn1 name number:', len(name_list))
        f2 = open(namefile2, 'r', encoding='utf-8')
        while 1:
            l = f2.readline().rstrip('\n')
            if l == '':
                break
            name = l+'2'
            name_list.append(name)
        f2.close()
        print('total name number:', len(name_list))
        name_list = list(set(name_list))
        print('number of different names:', len(name_list))
        self.emb_vec = self.model.get_emb(name_list)

    def save_embeddings(self, filename, vector):
        f = open(filename, 'w')
        for node, vec in vector.items():
            f.write("{} {}\n".format(node, ','.join([str(x) for x in vec])))
        f.close()
