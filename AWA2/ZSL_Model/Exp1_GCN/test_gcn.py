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


DATA_DIR_PREFIX = '/home/gyx/Data/KG_SS_DATA/ZSL/'
# DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'
EXP_NAME = 'AWA2/Exp1_GCN'


def test_imagenet_zero(fc_file_pred, has_train=False):
    with open(classids_file_retrain) as fp:  # corresp-awa.json
        classids = json.load(fp)

    with open(word2vec_file, 'rb') as fp:  # glove
        # word2vec_feat = pkl.load(fp, encoding='iso-8859-1')
        word2vec_feat = pkl.load(fp)

    testlist = []
    testlabels = []
    with open(vallist_folder) as fp:  # img-awa2.txt  测试数据文件
        for line in fp:
            fname, lbl = line.split()  # n03236735/n03236735_4047.JPEG 398

            assert int(lbl) >= 398
            # feat_name = os.path.join(feat_folder, fname.replace('.JPEG', '.mat'))
            feat_name = os.path.join(feat_folder, fname.replace('.jpg', '.npz'))  # 获取对应图片的特征文件

            if not os.path.exists(feat_name):
                print('not feature', feat_name)
                continue
            testlist.append(feat_name)
            testlabels.append(int(lbl))

    with open(fc_file_pred, 'rb') as fp:  # 模型文件
        # fc_layers_pred = pkl.load(fp, encoding='iso-8859-1')
        fc_layers_pred = pkl.load(fp)
    fc_layers_pred = np.array(fc_layers_pred)
    print('fc output', fc_layers_pred.shape)

    # remove invalid classes(wv = 0)
    valid_clss = np.zeros(22000)
    cnt_zero_wv = 0
    for j in range(len(classids)):
        if classids[j][1] == 1:
            twv = word2vec_feat[j]
            # print("twv:", twv)
            twv = twv / (np.linalg.norm(twv) + 1e-6)

            if np.linalg.norm(twv) == 0:
                cnt_zero_wv = cnt_zero_wv + 1
                continue
            valid_clss[classids[j][0]] = 1
            # print("valid_clss_value:", valid_clss[classids[j]])

    # process 'train' classes. they are possible candidates during inference
    cnt_zero_wv = 0
    labels_train, word2vec_train = [], []
    fc_now = []
    for j in range(len(classids)):
        tfc = fc_layers_pred[j]
        if has_train:
            if classids[j][0] < 0:
                continue
        else:
            if classids[j][1] == 0:
                continue

        if classids[j][0] >= 0:
            twv = word2vec_feat[j]
            # print("class ids:", classids[j][0])
            # print("train twv:", twv)
            if np.linalg.norm(twv) == 0:  # 求范数
                cnt_zero_wv = cnt_zero_wv + 1
                continue

            labels_train.append(classids[j][0])
            word2vec_train.append(twv)

            feat_len = len(tfc)
            tfc = tfc[feat_len - fc_dim: feat_len]

            fc_now.append(tfc)
    fc_now = np.array(fc_now)
    print('skip candidate class due to no word embedding: %d / %d:' % (cnt_zero_wv, len(labels_train) + cnt_zero_wv))
    print('candidate class shape: ', fc_now.shape)

    fc_now = fc_now.T
    labels_train = np.array(labels_train)
    print('train + test class: ', len(labels_train))

    # topKs = [1]
    # top_retrv = [1, 2, 5, 10, 20]
    # # top_retrv = [1]  # awa2 just top-1 accuracy
    # hit_count = np.zeros((len(topKs), len(top_retrv)))
    hit_count = 0

    cnt_valid = 0
    t = time.time()

    for j in range(len(testlist)):
        featname = testlist[j]
        # print("featname:", featname)
        # if valid_clss[testlabels[j]] == 0:
        #     continue

        cnt_valid = cnt_valid + 1

        # matfeat = sio.loadmat(featname)
        matfeat = np.load(featname)
        matfeat = matfeat['feat']

        scores = np.dot(matfeat, fc_now).squeeze()

        scores = scores - scores.max()
        scores = np.exp(scores)
        scores = scores / scores.sum()

        ids = np.argsort(-scores)

        for sort_id in range(1):
            lbl = labels_train[ids[sort_id]]

            if int(lbl) == testlabels[j]:
                hit_count = hit_count + 1
                break

        if j % 5000 == 0:
            inter = time.time() - t
            print('processing %d / %d ' % (j, len(testlist)), ', Estimated time: ',
                  inter / (j - 1) * (len(testlist) - j))

    hit_count = hit_count * 1.0 / cnt_valid

    # fout = open(fc_file_pred + '_result_pred_zero.txt', 'w')

    print(hit_count)
    print('total: %d', cnt_valid)
    # fout.write(outstr + '\n')
    # fout.close()

    return hit_count


# global varc
fc_dim = 0
wv_dim = 0

if __name__ == '__main__':


    model_path = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'glove_res50/output/feat_600')
    feat_folder = os.path.join(DATA_DIR_PREFIX, 'AWA2/Test_DATA_feats/')

    fc_dim = 2048
    word2vec_file = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'glove_word2vec_wordnet.pkl')



    vallist_folder = os.path.join(DATA_DIR_PREFIX, 'AWA2/materials', 'test_img_list_awa.txt')
    classids_file_retrain = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'corresp-train-test-awa.json')

    print('\nEvaluating ...\nPlease be patient for it takes a few minutes...')

    res = test_imagenet_zero(fc_file_pred=model_path)
    output = res

    print('----------------------')
    print('model : ', model_path)
    print('result: ', output)
    print('----------------------')


