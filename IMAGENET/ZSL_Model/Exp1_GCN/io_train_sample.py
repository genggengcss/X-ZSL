# coding=gbk
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import pickle as pkl
from scipy import sparse
import os

from tensorflow.python import pywrap_tensorflow

from io_graph import prepare_graph
'''
original:convert_to_gcn_data.py
'''


DATA_DIR_PREFIX = '/home/gyx/Data/KG_SS_DATA/ZSL/'
# DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'
EXP_NAME = 'IMAGENET/Exp1_GCN'

# save embedding vectors of all the vertices of the graph
def convert_input(wv_file, save_dir):
    with open(wv_file, 'rb') as fp:
        vertex_vectors = pkl.load(fp)
    vertex_vectors = vertex_vectors.tolist()   # 数组——>列表
    sparse_vecs = sparse.csr_matrix(vertex_vectors)  # 压缩矩阵稀疏行
    dense_vecs = np.array(vertex_vectors)

    sparse_file = os.path.join(save_dir, 'all_x.pkl')
    with open(sparse_file, 'wb') as fp:
        pkl.dump(sparse_vecs, fp)

    dense_file = os.path.join(save_dir, 'all_x_dense.pkl')  # embedding vectors of all vertices
    with open(dense_file, 'wb') as fp:
        pkl.dump(dense_vecs, fp)

    print('Save vectors of all vertices')


def convert_label(model_path, layer_name, save_dir):  # get output's label and mask
    ''' save visual classifier '''
    corresp_file = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'corresp-train-test.json')
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

    # the position of a seen class has the vector of corresponding CNN feature weight
    # fc[:, class_id + offset] represents the feature weights of class_id
    fc_labels = np.zeros((len(corresp_list), fc_dim))
    print('fc dim ', fc_labels.shape)
    for i, corresp in enumerate(corresp_list):
        vertex_type = corresp[1]
        class_id = corresp[0]
        # seen class (vertex)
        if vertex_type == 0:
            fc_labels[i, :] = np.copy(fc[:, class_id])
            assert class_id < 1000
    label_file = os.path.join(save_dir, 'train_y.pkl')   #  和all_a_dense.pkl同样的行数，seen classes的feature wegiths写到对应的行，其它的行为0；
    with open(label_file, 'wb') as fp:
        pkl.dump(fc_labels, fp)

    # the position that is 1 means the vertex of that position is an unseen class
    test_index = []
    for corresp in corresp_list:
        if corresp[0] == -1:
            test_index.append(-1)
        else:
            test_index.append(corresp[1])  # corresp[1]: 0/1, the value 0 means seen class, 1 means unseen classes
    test_file = os.path.join(save_dir, 'test_index.pkl')  # 和all_a_dense.pkl同样的行数
    with open(test_file, 'wb') as fp:
        pkl.dump(test_index, fp)


def convert_graph(save_dir):
    graph_file = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'imagenet_graph.pkl')
    if not os.path.exists(graph_file):
        prepare_graph()
    save_file = os.path.join(save_dir, 'graph.pkl')
    if os.path.exists(save_file):
        cmd = 'rm  %s' % save_file
        os.system(cmd)
    cmd = 'ln -s %s %s' % (graph_file, save_file)  # 软链接，将 原来的imagenet_graph.pkl，链接到目标文件夹下
    os.system(cmd)


if __name__ == '__main__':

    model_path = os.path.join(DATA_DIR_PREFIX, 'materials', 'resnet_v1_50.ckpt')
    layer_name = 'resnet_v1_50/logits/weights'

    wv_file = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'glove_word2vec_wordnet.pkl')

    save_dir = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'glove_res50')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('Converting input')
    convert_input(wv_file, save_dir)

    print('Converting graph')
    convert_graph(save_dir) # not need soft link, graph file is imagenet_graph.pkl

    print('Converting label')
    convert_label(model_path, layer_name, save_dir)
    print('Prepared data to %s' % save_dir)
