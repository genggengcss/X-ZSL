# coding=gbk
# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf
import pickle as pkl
import os
import numpy as np

import sys
# sys.path.append('../../../../../')
# from ZSL.IMAGENET_Animal.Exp_Test.utils import *




# DATA_DIR_PREFIX = '/Users/geng/Data/Human_X_ZSL_DATA/'
DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'

EXP_NAME = 'IMAGENET_Animal/Exp_DeVise'

Testlist_Fils_Less = os.path.join(DATA_DIR_PREFIX, 'IMAGENET_Animal/materials', 'test_img_list_2_hops_animal.txt')
test_img_feat_folder = os.path.join(DATA_DIR_PREFIX, 'Test_DATA_feats/Test_DATA_feats/')  # img features of test images


def add_layer(inputs, in_size, out_size, activation_function=None,):  # 默认为None时，表示为线性函数
    # add one more layer and return the output of this layer

    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 随机生成一个变量矩阵，且符合正态分布
    # biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)  # 1行 out_size列，初始值推荐不为0
    # Wx_plus_b = tf.matmul(inputs, Weights)+biases
    Wx_plus = tf.matmul(inputs, Weights)
    if activation_function:
        outputs = tf.nn.l2_normalize(Wx_plus, dim=1)
    else:
        outputs = Wx_plus
    return outputs


def model_test(sess, test_y_label, test_y_index):
    global preds

    test_feat_file_path = []
    testlabels = []
    with open(Testlist_Fils_Less) as fp:  # test_image_list.txt  测试数据文件
        for line in fp:
            fname, lbl = line.split()  # n03236735/n03236735_4047.JPEG 398

            assert int(lbl) >= 398
            # feat_name = os.path.join(feat_folder, fname.replace('.JPEG', '.mat'))
            feat_name = os.path.join(test_img_feat_folder, fname.replace('.JPEG', '.npz'))  # 获取对应图片的特征文件

            if not os.path.exists(feat_name):
                print('not feature', feat_name)
                continue
            test_feat_file_path.append(feat_name)
            testlabels.append(int(lbl))

    test_y_label = test_y_label.T

    ## imagenet 2-hops topK result
    topKs = [1]
    top_retrv = [1, 2, 5, 10, 20]
    hit_count = np.zeros((len(topKs), len(top_retrv)))

    # hit_count = 0
    cnt_valid = 0  # count test images


    for j in range(len(testlabels)):
        test_feat_file = test_feat_file_path[j]

        if valid_class[testlabels[j]] == 0:  # remove invalid unseen classes
            continue

        cnt_valid = cnt_valid + 1

        test_feat = np.load(test_feat_file)
        test_feat = test_feat['feat']  # [2048]

        test_feat = test_feat[np.newaxis, :]  # [1, 2048]

        y_pre = sess.run(preds, feed_dict={xs: test_feat})  # [1, 300]

        scores = np.dot(y_pre, test_y_label).squeeze()

        scores = scores - scores.max()
        scores = np.exp(scores)
        scores = scores / scores.sum()

        ids = np.argsort(-scores)

        for top in range(len(topKs)):
            for k in range(len(top_retrv)):
                current_len = top_retrv[k]

                for sort_id in range(current_len):
                    lbl = test_y_index[ids[sort_id]]
                    # print("predicted label:", lbl)
                    # print("ground truth label:", testlabels[j])
                    if int(lbl) == testlabels[j]:
                        hit_count[top][k] = hit_count[top][k] + 1
                        break

        if j % 10000 == 0:

            print('processing %d / %d ' % (j, len(testlabels)))

    hit_count = hit_count * 1.0 / cnt_valid

    return hit_count








def readPKL(file_name):
    with open(file_name, 'rb') as fp:  # glove
        feat = pkl.load(fp)
    print("feat shape:", feat.shape)
    return feat





# load data
glove_file = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'glove_word2vec.pkl')
x_file = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'train_y_398.pkl')

x_feat = readPKL(x_file)  # [398, 2048]
y_feat = readPKL(glove_file) # [895, 300]


# remove invalid training class
invalid_wv = 0
train_x, train_y = [], []
for k in range(0, 398):
    t_wv = y_feat[k]
    if np.linalg.norm(t_wv) == 0:  # 求范数
        invalid_wv = invalid_wv + 1
        continue
    train_y.append(t_wv)
    train_x.append(x_feat[k])

train_x = np.array(train_x)  # [398, 2048]
train_y = np.array(train_y)  # [398, 300]
#  8 / 398
print('skip seen class due to no word embedding: %d / %d:' % (invalid_wv, len(train_y) + invalid_wv))


# remove invalid testing class
test_y_label = []
test_y_index = []
# remove invalid unseen classes(wv = 0)
valid_class = np.zeros(22000)
invalid_unseen_wv = 0
for j in range(398, len(y_feat)):
    t_wv = y_feat[j]
    t_wv = t_wv / (np.linalg.norm(t_wv) + 1e-6)

    if np.linalg.norm(t_wv) == 0:
        invalid_unseen_wv = invalid_unseen_wv + 1
        continue
    valid_class[j] = 1
    test_y_label.append(t_wv)
    test_y_index.append(j)

# 26/ 497 = 471
print('skip unseeen class due to no word embedding: %d / %d:' % (invalid_unseen_wv, len(test_y_label)+invalid_unseen_wv))
test_y_label = np.array(test_y_label)
test_y_index = np.array(test_y_index)






# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 2048])  # feature embedding
ys = tf.placeholder(tf.float32, [None, 300])  # glove embedding

# 定义输出层
preds = add_layer(xs, 2048, 300, activation_function=True)  # 输出 dim(xs) * 10 的向量
# compute loss
preds = preds * 10
y_label = tf.nn.l2_normalize(ys, dim=1)
y_label = y_label * 10
loss = tf.nn.l2_loss(tf.subtract(y_label, preds))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


for i in range(2000):

    sess.run(train_step, feed_dict={xs: train_x, ys: train_y})

    if i % 20 == 0:
        looss = sess.run(loss, feed_dict={xs: train_x, ys: train_y})
        print("epoch :", i, ", loss :", looss)



    if i >= 500 and i % 100 == 0:
        result = model_test(sess, test_y_label, test_y_index)

        output = ['{:.2f}'.format(i * 100) for i in result[0]]
        print("epoch :", output)
