# coding=gbk
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import pickle as pkl
from scipy import sparse
import os

from tensorflow.python import pywrap_tensorflow
import tensorflow as tf

import torch
import torch.nn.functional as F

from io_graph import prepare_graph
'''
convert to gcn data: input, output, graph
'''

# DATA_DIR_PREFIX = '/home/gyx/Data/KG_SS_DATA/ZSL/'
DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'
EXP_NAME = 'AWA2/Exp1_GCN'

def consin_s(x3, x4, axis):
    # 求模
    x3_norm = tf.sqrt(tf.reduce_sum(tf.square(x3), axis=axis))
    x4_norm = tf.sqrt(tf.reduce_sum(tf.square(x4), axis=axis))
    # 内积
    x3_x4 = tf.reduce_sum(tf.multiply(x3, x4), axis=axis)
    consin = tf.divide(x3_x4, tf.multiply(x3_norm, x4_norm))
    return consin


def add_atten_cos(inputs):

    output = inputs
    print output

    # consin distance
    output = F.normalize(output)
    output_T = output.t()
    logits = torch.mm(output, output_T)

    coefs = F.softmax(logits, dim=1)
    # coefs = logits
    # print coefs.shape
    print coefs

def convert_label(model_path, layer_name, save_dir):  # get output's label and mask
    ''' save visual classifier '''
    corresp_file = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'corresp-train-test-awa.json')
    with open(corresp_file) as fp:
        corresp_list = json.load(fp)

    def get_variables_in_checkpoint_file(file_name):
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)  # 利用pywrap_tensorflow获取ckpt文件中的所有变量，得到的是variable名字与shape的一个map
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map, reader

    # read feature weights for seen classes, ordered by class id
    var_keep_dic, reader = get_variables_in_checkpoint_file(model_path)
    for name in var_keep_dic:
        print(name, len(var_keep_dic[name]), var_keep_dic[name])
        if name == layer_name:
            print(name)
            print(reader.get_tensor(name).shape)
            fc = reader.get_tensor(name).squeeze()
            fc_dim = fc.shape[0]
            break
    # print('fc:', fc.shape)  # (2048, 1000)

    fc_labels = np.transpose(fc)
    fc_labels = fc_labels[0:398]
    fc_labels = torch.from_numpy(fc_labels)
    print('fc:', fc_labels.shape)
    add_atten_cos(fc_labels)


    # D = []
    # for i in range(len(fc_labels)):
    #     d = []
    #     for j in range(len(fc_labels)):
    #         # e1 = tf.sqrt(tf.reduce_sum(tf.square(x3[i] - x4[j]), 0))
    #         e1 = consin_s(fc_labels[i], fc_labels[j], 0)
    #         d.append(e1)
    #     D.append(d)
    # Dt = tf.convert_to_tensor(D)
    #
    # # compute the cosine distance simlarity
    # one = tf.ones([1, 2048], tf.float32)
    # tff = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(fc_labels), 1)), 1)
    # norm = tf.matmul(tff, one)
    # outputs_select_nor = tf.divide(fc_labels, norm)
    # logits = tf.matmul(outputs_select_nor, tf.transpose(outputs_select_nor))
    #
    # with tf.Session() as sess:
    #     print sess.run(logits)
    # # the position of a seen class has the vector of corresponding CNN feature weight
    # # fc[:, class_id] represents the feature weights of class_id
    # fc_labels = np.zeros((len(corresp_list), fc_dim))
    # print('fc dim ', fc_labels.shape)
    # for i, corresp in enumerate(corresp_list):
    #     vertex_type = corresp[1]
    #     class_id = corresp[0]
    #     # seen class (vertex)
    #     if vertex_type == 0:
    #         fc_labels[i, :] = np.copy(fc[:, class_id])
    #         assert class_id < 398
    # # label_file = os.path.join(save_dir, 'train_y.pkl')   #  和all_a_dense.pkl同样的行数，seen classes的feature wegiths写到对应的行，其它的行为0；
    # # with open(label_file, 'wb') as fp:
    # #     pkl.dump(fc_labels, fp)
    #
    # print 'fc_label shape:', fc_labels.shape





if __name__ == '__main__':




    model_path = os.path.join(DATA_DIR_PREFIX, 'materials', 'resnet_v1_50.ckpt')
    layer_name = 'resnet_v1_50/logits/weights'


    save_dir = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'glove_res50')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    print('Converting label')
    convert_label(model_path, layer_name, save_dir)
    print('Prepared data to %s' % save_dir)
