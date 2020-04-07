from __future__ import print_function
import argparse
import json
import numpy as np
import os

import pickle as pkl
import scipy.io as sio
import time
''' 
test total accuracy of attentive gcn model
'''




def val(weight_pred, dir, dataset):

    PATH = os.path.join(dir, dataset)
    feat_dim = 2048  # the dimension of CNN features
    # the word embedidngs of graph nodes, to ensure that the testing classes have initialization
    word2vec_file = os.path.join(PATH, 'glove_w2v.pkl')
    # mark the type of graph nodes (seen, unseen, others)
    classids_file_retrain = os.path.join(PATH, 'corresp.json')

    # testing image list
    # testlist_folder = os.path.join(DATA_DIR, DATASET, 'test_img_list_100.txt')
    val_list_folder = os.path.join(PATH, 'val_img_list.txt')

    # img features of test images
    val_img_feat_folder = os.path.join(PATH, 'Val_DATA_feats')



    test_feat_file_path = []
    testlabels = []
    with open(val_list_folder) as fp:  # test_image_list.txt
        for line in fp:
            fname, lbl = line.split()  # n03236735/n03236735_4047.JPEG 398 [image path, image label]

            assert int(lbl) >= 398
            # images of different dataset have different suffix
            if dataset == 'ImNet_A':
                feat_name = os.path.join(val_img_feat_folder, fname.replace('.JPEG', '.npz'))  # get the image features
            if dataset == 'AwA':
                feat_name = os.path.join(val_img_feat_folder, fname.replace('.jpg', '.npz'))  # get the image features

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

    # obtain predicted (unseen) classifiers

    weight_pred = np.array(weight_pred)

    # print('weight_pred output', weight_pred.shape)



    invalid_wv = 0  # count the number of invalid class embedding
    labels_testval, word2vec_testval = [], []  # zsl: unseen label and its class embedding
    weight_pred_testval = []  # zsl: unseen classifier
    for j in range(len(classids)):
        t_wpt = weight_pred[j]
        # just process the unseen class
        if classids[j][1] == 0:
            continue

        if classids[j][0] >= 0:
            t_wv = word2vec_feat[j]
            # if the word embedding is valid
            if np.linalg.norm(t_wv) == 0:
                invalid_wv = invalid_wv + 1
                continue
            labels_testval.append(classids[j][0])
            word2vec_testval.append(t_wv)

            feat_len = len(t_wpt)
            t_wpt = t_wpt[feat_len - feat_dim: feat_len]
            weight_pred_testval.append(t_wpt)
    weight_pred_testval = np.array(weight_pred_testval)
    # print('skip candidate class due to no word embedding: %d / %d:' % (invalid_wv, len(labels_testval) + invalid_wv))
    # print('candidate class shape: ', weight_pred_testval.shape)

    weight_pred_testval = weight_pred_testval.T
    labels_testval = np.array(labels_testval)
    # print('final test classes: ', len(labels_testval))

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


    # topK result
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



    hit_count = hit_count * 1.0 / cnt_valid
    # print(hit_count)
    # print('total: %d', cnt_valid)

    return hit_count



