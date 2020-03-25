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
from collections import Counter

# DATA_DIR_PREFIX = '/home/gyx/Data/KG_SS_DATA/ZSL/'
DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'

DATA_DIR = 'AWA2/materials/Attributes'


class_att_table = os.path.join(DATA_DIR_PREFIX, DATA_DIR, 'class-attribute-table.txt')

class_att_dict = dict()
with open(class_att_table, 'r') as fp:
    for line in fp.readlines():
        line = line.strip()
        lines = line.split('\t')
        class_att_dict[lines[0]] = lines[1:]
print(class_att_dict)

# print(class_att_dict['killer+whale'])
# print(class_att_dict['dolphin'])

seen_att = class_att_dict['zebra']
unseen_att = class_att_dict['horse']

# seen_att = class_att_dict['killer+whale']
# unseen_att = class_att_dict['dolphin']

# seen_att = class_att_dict['humpback+whale']
# unseen_att = class_att_dict['blue+whale']

# polar+bear
# grizzly+bear
# raccoon

print(len(unseen_att))
print(len(seen_att))



overlap = list()
overall = list()

for att in seen_att:
    overall.append(att)

for att in unseen_att:
    if att in seen_att:
        overlap.append(att)
        continue
    overall.append(att)


print(overlap)
print("overlap:", len(overlap))
print("overall:", len(overall))

unseen_att = class_att_dict['rat']
seen_att1 = class_att_dict['beaver']
seen_att2 = class_att_dict['hamster']
# seen_att3 = class_att_dict['squirrel']
seen_att4 = class_att_dict['mouse']

print(len(unseen_att))
print(len(seen_att1))
print(len(seen_att2))
# print(len(seen_att3))
print(len(seen_att4))

over = list()
for att in unseen_att:
    if att in seen_att1 and att in seen_att2 and att in seen_att4:
        over.append(att)
print('overlap:', len(over))

over2 = list()
for att in seen_att1:
    if att in seen_att2 and att in seen_att4:
        over.append(att)
print('overlap2:', len(over))

class_att = list()
class_att.extend(unseen_att)
class_att.extend(seen_att1)
class_att.extend(seen_att2)
# class_att.extend(seen_att3)
class_att.extend(seen_att4)
print len(class_att)
print len(set(class_att))

c = Counter(class_att)
print len(c)
print c


unseen_att = class_att_dict['zebra']
seen_att1 = class_att_dict['horse']
# seen_att2 = class_att_dict['polar+bear']


print(len(unseen_att))
print(len(seen_att1))
# print(len(seen_att2))

class_att = list()
class_att.extend(unseen_att)
class_att.extend(seen_att1)
# class_att.extend(seen_att2)

print len(class_att)

c = Counter(class_att)
print len(c)
print c

over = list()
for att in unseen_att:
    # if att in seen_att1 and att in seen_att2:
    if att in seen_att1:

        over.append(att)
print('overlap:', len(over))

# print over


# # unseen_att = class_att_dict['grizzly+bear']
# # seen_att1 = class_att_dict['raccoon']
# seen_att2 = class_att_dict['shark']
#
#
# # print(len(unseen_att))
# # print(len(seen_att1))
# print(seen_att2)