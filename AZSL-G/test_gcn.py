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
'''
testing file 
test the total accuracy of gcn model
'''




def test_imagenet_zero(weight_pred_file, has_train=False):
    test_feat_file_path = []
    testlabels = []
    print('test file: ....', testlist_folder)
    with open(testlist_folder) as fp:  # test_image_list.txt  测试数据文件
        for line in fp:
            fname, lbl = line.split()  # n03236735/n03236735_4047.JPEG 398

            assert int(lbl) >= 398
            # feat_name = os.path.join(feat_folder, fname.replace('.JPEG', '.mat'))
            if DATASET == 'ImNet_A':
                feat_name = os.path.join(test_img_feat_folder, fname.replace('.JPEG', '.npz'))  # 获取对应图片的特征文
            if DATASET == 'AwA':
                feat_name = os.path.join(test_img_feat_folder, fname.replace('.jpg', '.npz'))  # 获取对应图片的特征文件

            if not os.path.exists(feat_name):
                print('no feature', feat_name)
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
    top_retrv = [1, 2, 3, 5, 10, 20]
    hit_count = np.zeros((len(topKs), len(top_retrv)))

    # hit_count = 0
    cnt_valid = 0  # count test images
    t = time.time()
    lbl_list = list()
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
                    lbl_list.append(int(lbl))
                    if int(lbl) == testlabels[j]:
                        hit_count[top][k] = hit_count[top][k] + 1
                        break

        if j % 10000 == 0:
            inter = time.time() - t
            print('processing %d / %d ' % (j, len(testlabels)), ', Estimated time: ',
                  inter / (j - 1) * (len(testlabels) - j))

    # save to file
    lab = 'label.txt'
    wr_fp = open(lab, 'w')
    for lal in lbl_list:
        wr_fp.write('%s\n' % lal)
    wr_fp.close()

    hit_count = hit_count * 1.0 / cnt_valid
    # print(hit_count)
    print('total: %d', cnt_valid)
    return hit_count




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='/home/gyx/X_ZSL/data', help='root directory')
    parser.add_argument('--data_dir', type=str, default='GCNZ', help='data directory')
    parser.add_argument('--dataset', type=str, default='ImNet_A', help='ImNet_A, AwA')
    parser.add_argument('--feat', type=str, default='', help='the predicted file')

    args = parser.parse_args()

    DATA_DIR = os.path.join(args.data_root, args.data_dir)
    DATASET = args.dataset
    EXP_NAME = 'Exp1_GCN'


    feat_dim = 2048
    word2vec_file = os.path.join(DATA_DIR, DATASET, 'glove_w2v.pkl')
    classids_file_retrain = os.path.join(DATA_DIR, DATASET, 'corresp.json')
    # testlist_folder = os.path.join(DATA_DIR, DATASET, 'test_img_list_100.txt')
    testlist_folder = os.path.join(DATA_DIR, DATASET, 'test_img_list.txt')

    test_img_feat_folder = os.path.join(DATA_DIR, DATASET, 'Test_DATA_feats')  # img features of test images

    # ImNet_A: python test_gcn.py --feat 500
    # AwA: python test_gcn.py --feat 500

    training_outputs = os.path.join(DATA_DIR, DATASET, EXP_NAME, 'feat_' + args.feat)


    print('\nEvaluating ...\nPlease be patient for it takes a few minutes...')

    res = test_imagenet_zero(weight_pred_file=training_outputs)

    output = ['{:.2f}'.format(i * 100) for i in res[0]]


    print('----------------------')
    print('model : ', training_outputs)
    print('result: ', output)
    print('----------------------')


