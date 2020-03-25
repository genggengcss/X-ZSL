# coding=gbk
# -*- coding: utf-8 -*-
'''
extract feature of test images
original: extract_pool5.py
'''
import argparse
import os
import numpy as np
import cv2

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets import inception_v1
from tensorflow.contrib.slim.python.slim.nets.inception_v1 import inception_v1_base
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.python.ops import variable_scope
import time


# DATA_DIR_PREFIX = '/home/gyx/Data/KG_SS_DATA/ZSL/'
DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'

DATA_DIR = 'AWA2/materials/Attributes'

class_file = os.path.join(DATA_DIR_PREFIX, DATA_DIR, 'classes.txt')
att_file = os.path.join(DATA_DIR_PREFIX, DATA_DIR, 'predicates.txt')
class_att_file = os.path.join(DATA_DIR_PREFIX, DATA_DIR, 'predicate-matrix-binary.txt')
class_att_table = os.path.join(DATA_DIR_PREFIX, DATA_DIR, 'class-attribute-table.txt')

id_class_list = list()
with open(class_file, 'r') as fp:
    for line in fp.readlines():
        line = line.strip()
        index, class_name = line.split('\t')
        id_class_list.append(class_name)
print(id_class_list)

id_att_list = list()
with open(att_file, 'r') as fp:
    for line in fp.readlines():
        line = line.strip()
        index, att_name = line.split('\t')
        id_att_list.append(att_name)
print(id_att_list)


class_att_list = list()
with open(class_att_file, 'r') as fp:
    index = 0
    for line in fp.readlines():
        line = line.strip()
        atts = line.split(' ')
        atts = map(int, atts)  # python2: string -> int
        att_nonzero = np.nonzero(atts)

        line = id_class_list[index]
        for att_index in att_nonzero[0]:
            att_name = id_att_list[att_index]
            line = line + '\t' + att_name
        print line
        class_att_list.append(line)
        index = index + 1

# print class_att_list


wr_fp = open(class_att_table, 'w')
for line in class_att_list:
    wr_fp.write('%s\n' % line)

wr_fp.close()




