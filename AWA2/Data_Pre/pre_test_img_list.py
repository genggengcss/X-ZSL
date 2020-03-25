# coding=gbk
# -*- coding: utf-8 -*-

import argparse
import os
import threading
import urllib
import glob

import cv2
import numpy as np
'''
download original images
'''


IMAGE_SAVE_DIR = '/home/gyx/Data/Ori_DATA/AwA2data/'  # save directory
# DATA_DIR_PREFIX = 'home/gyx/Data/KG_SS_DATA/ZSL/
DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'

list_file = os.path.join(DATA_DIR_PREFIX, 'materials', 'unseen_awa2.txt')
save_file = os.path.join(DATA_DIR_PREFIX, 'materials', 'test_image_list_awa.txt')

# create the list file of testing images and labels
def make_image_list(offset):
    with open(list_file) as fp:
        wnid_list = [line.strip() for line in fp]
    print(len(wnid_list))


    wr_fp = open(save_file, 'w')
    for i, wnid in enumerate(wnid_list):
        img_list = glob.glob(os.path.join(IMAGE_SAVE_DIR, wnid, '*.jpg'))
        for path in img_list:
            index = os.path.join(wnid, os.path.basename(path))
            l = i + offset
            wr_fp.write('%s %d\n' % (index, l))
        if len(img_list) == 0:
            print('Warning: does not have class %s. Do you forgot to download the picture??' % wnid)
    wr_fp.close()




if __name__ == '__main__':

    offset = 398

    make_image_list(offset)

