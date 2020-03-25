# coding=gbk
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import json
import numpy as np
import os
'''
function for reading file (txt, json, pkl...)
'''
import pickle as pkl
# DATA_DIR = '/Users/geng/Data/Human_X_ZSL_DATA/'
DATA_DIR = '/home/gyx/Data/Exp_DATA/Human_X_ZSL_DATA/'
EXP_DATA = 'Exp1_GCN'

# global dir variable
feat_dim = 2048
Word2vec_File = os.path.join(DATA_DIR, EXP_DATA, 'glove_word2vec_wordnet.pkl')
Classids_File_Retrain = os.path.join(DATA_DIR, EXP_DATA, 'corresp-train-test-animal.json')
Test_Img_Feat_Folder = os.path.join(DATA_DIR, 'Test_DATA', 'imgnet_2_hops_test_feats/')  # img features of test images
Testlist_File = os.path.join(DATA_DIR, 'materials', 'test_image_list_2_hops.txt')


# other files
inv_wordn_file = os.path.join(DATA_DIR, EXP_DATA, 'invdict_wordn.json')  # wnids in graph index
train_animal_nodes_file = os.path.join(DATA_DIR, 'materials', 'train_animal.txt')
test_animal_nodes_file = os.path.join(DATA_DIR, 'materials', 'test_animal_2_hops.txt')
wnid_wname_file = os.path.join(DATA_DIR, 'materials', 'words_animal.txt')


# read the file - wnid index in graph
def obtain_wnid_in_graph_index():
    with open(inv_wordn_file) as fp:
        wnids = json.load(fp)
    return wnids

# read the file - wnid of train/test set
def obtain_wnid(wnid_file):
    # animal train
    animal = []
    nodes = open(wnid_file, 'rU')
    try:
        for line in nodes:
            line = line[:-1]  # 直接 输出的 line 带"\n"，去除最后的 "\n"
            animal.append(line)
    finally:
        nodes.close()
    return animal

# read the dictionary file - wnid and its text name
def obtain_wnid_wname_dict():
    words = dict()
    with open(wnid_wname_file, 'r') as fp:
        for line in fp.readlines():
            wnid, name = line.split('\t')
            wnid = wnid.strip()
            name = name.strip()
            words[wnid] = name
    return words


TRAIN_WNID = obtain_wnid(train_animal_nodes_file)  # training set wnid
TEST_WNID = obtain_wnid(test_animal_nodes_file)  # testing set wnid
WNIDS_IN_GRAPH_INDEX = obtain_wnid_in_graph_index()  # wnid index in graph
WNID_WNAME = obtain_wnid_wname_dict()  # wnid and its word name