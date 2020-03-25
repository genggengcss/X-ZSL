# coding=gbk
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import json
import numpy as np
import os

import pickle as pkl
import scipy.io as sio
import time
''' original testing file 
'''

# DATA_DIR_PREFIX = '/home/gyx/Data/KG_SS_DATA/ZSL/'
DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'
EXP_NAME = 'IMAGENET/Exp1_GCN'

def test_imagenet_zero(weight_pred_file, has_train=False):


    test_feat_file_path = []
    testlabels = []
    with open(testlist_folder) as fp:  # imgnet_2_hops_test_img.txt  测试数据文件
        for line in fp:
            fname, lbl = line.split()  # n03236735/n03236735_4047.JPEG 1000

            assert int(lbl) >= 1000
            # feat_name = os.path.join(feat_folder, fname.replace('.JPEG', '.mat'))
            feat_name = os.path.join(test_img_feat_folder, fname.replace('.JPEG', '.npz'))  # 获取对应图片的特征文件

            if not os.path.exists(feat_name):
                print('not feature', feat_name)
                continue
            test_feat_file_path.append(feat_name)
            testlabels.append(int(lbl))

    with open(classids_file_retrain) as fp:  # corresp-awa.json
        classids = json.load(fp)
    with open(word2vec_file, 'rb') as fp:  # glove
        # word2vec_feat = pkl.load(fp, encoding='iso-8859-1')
        word2vec_feat = pkl.load(fp)

    # obtain training results
    with open(weight_pred_file, 'rb') as fp:  #
        weight_pred = pkl.load(fp)
    weight_pred = np.array(weight_pred)

    print('weight_pred output', weight_pred.shape)


    # process 'train' classes. they are possible candidates during inference
    invalid_wv = 0  # count the number of invalid class embedding
    labels_testval, word2vec_testval = [], []  # zsl: unseen label and its class embedding
    weight_pred_testval = []  # zsl: unseen output feature
    for j in range(len(classids)):
        t_wpt = weight_pred[j]
        if has_train:
            if classids[j][0] < 0:
                continue
        else:
            if classids[j][1] == 0:
                continue

        if classids[j][0] >= 0:
            t_wv = word2vec_feat[j]
            if np.linalg.norm(t_wv) == 0:  # 求范数
                invalid_wv = invalid_wv + 1
                continue
            labels_testval.append(classids[j][0])
            word2vec_testval.append(t_wv)

            feat_len = len(t_wpt)
            t_wpt = t_wpt[feat_len - feat_dim: feat_len]
            weight_pred_testval.append(t_wpt)
    weight_pred_testval = np.array(weight_pred_testval)
    print('skip candidate class due to no word embedding: %d / %d:' % (invalid_wv, len(labels_testval) + invalid_wv))
    print('candidate class shape: ', weight_pred_testval.shape)

    weight_pred_testval = weight_pred_testval.T
    labels_testval = np.array(labels_testval)
    print('final test classes: ', len(labels_testval))

    # remove invalid unseen classes(wv = 0)
    valid_class = np.zeros(22000)
    invalid_unseen_wv = 0
    for j in range(len(classids)):
        if classids[j][1] == 1:  # unseen classes
            t_wv = word2vec_feat[j]
            t_wv = t_wv / (np.linalg.norm(t_wv) + 1e-6)

            if np.linalg.norm(t_wv) == 0:
                invalid_unseen_wv = invalid_unseen_wv + 1
                continue
            valid_class[classids[j][0]] = 1


    ## imagenet 2-hops topK result
    topKs = [1]
    top_retrv = [1, 2, 5, 10, 20]
    hit_count = np.zeros((len(topKs), len(top_retrv)))

    # hit_count = 0
    cnt_valid = 0  # count test images
    t = time.time()

    for j in range(len(testlabels)):
        test_feat_file = test_feat_file_path[j]

        if valid_class[testlabels[j]] == 0:   # remove invalid unseen classes
            continue

        cnt_valid = cnt_valid + 1

        test_feat = np.load(test_feat_file)
        test_feat = test_feat['feat']

        scores = np.dot(test_feat, weight_pred_testval).squeeze()

        scores = scores - scores.max()
        scores = np.exp(scores)
        scores = scores / scores.sum()

        ids = np.argsort(-scores)

        for top in range(len(topKs)):
            for k in range(len(top_retrv)):
                current_len = top_retrv[k]

                for sort_id in range(current_len):
                    lbl = labels_testval[ids[sort_id]]
                    if int(lbl) == testlabels[j]:
                        hit_count[top][k] = hit_count[top][k] + 1
                        break

        if j % 10000 == 0:
            inter = time.time() - t
            print('processing %d / %d ' % (j, len(testlabels)), ', Estimated time: ',
                  inter / (j - 1) * (len(testlabels) - j))

    hit_count = hit_count * 1.0 / cnt_valid
    # print(hit_count)
    print('total: %d', cnt_valid)
    return hit_count


# global varc
feat_dim = 2048
word2vec_file = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'glove_word2vec_wordnet.pkl')
# inv_wordn_file = os.path.join(DATA_DIR, EXP_NAME, 'invdict_wordn.json')
classids_file_retrain = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'corresp-train-test.json')
testlist_folder = os.path.join(DATA_DIR_PREFIX, 'IMAGENET/materials', 'test_img_list_2_hops.txt')
test_img_feat_folder = os.path.join(DATA_DIR_PREFIX, 'IMAGENET/Test_DATA_feats/')  # img features of test images


if __name__ == '__main__':

    training_outputs = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'glove_res50/output/feat_400')
    # training_coefs = os.path.join(DATA_DIR, EXP_NAME, 'glove_res50_ani/output_atten/feat_atten_300')
    # wnids = obtain_wnid_in_graph(inv_wordn_file)



    print('\nEvaluating ...\nPlease be patient for it takes a few minutes...')

    res = test_imagenet_zero(weight_pred_file=training_outputs)

    output = ['{:.2f}'.format(i * 100) for i in res[0]]


    print('----------------------')
    print('model : ', training_outputs)
    print('result: ', output)
    print('----------------------')


