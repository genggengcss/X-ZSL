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



# DATA_DIR_PREFIX = '/Users/geng/Data/Human_X_ZSL_DATA/'
DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'
EXP_DATA = 'IMAGENET_Animal/Exp1_GCN'
EXP_NAME = 'IMAGENET_Animal/Exp_ConSE'

train_animal_nodes_file = os.path.join(DATA_DIR_PREFIX, 'IMAGENET/materials', 'seen_2012_1k.txt')
test_animal_nodes_file = os.path.join(DATA_DIR_PREFIX, 'IMAGENET_Animal/materials', 'unseen_2012_2-hops_animal.txt')
wnid_wname_file = os.path.join(DATA_DIR_PREFIX, 'materials', 'words.txt')

class_file = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'animal_class_name.json')


# read the file - wnid of train/test set
def obtain_wnid(train_file, test_file):
    # animal train
    animal = []
    train_nodes = open(train_file, 'rU')
    try:
        for line in train_nodes:
            line = line[:-1]  # 直接 输出的 line 带"\n"，去除最后的 "\n"
            animal.append(line)
    finally:
        train_nodes.close()

    test_nodes = open(test_file, 'rU')
    try:
        for line in test_nodes:
            line = line[:-1]  # 直接 输出的 line 带"\n"，去除最后的 "\n"
            animal.append(line)
    finally:
        test_nodes.close()
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


WNIDs = obtain_wnid(train_animal_nodes_file, test_animal_nodes_file)  # training+testing set wnid
print(len(WNIDs))
WNID_WNAME = obtain_wnid_wname_dict()  # wnid and its word name


# get and save words of each vertex
names = []
for wnid in WNIDs:
    if wnid in WNID_WNAME:
        names.append(WNID_WNAME[wnid])
        # print(WNID_WNAME[wnid])
    else:
        names.append('--%s--' % wnid)
        print('%s has no words' % wnid)

with open(class_file, 'w') as fp:
    json.dump(names, fp)
    print('Save graph node in text to %s' % class_file)