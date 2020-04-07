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


def test_imagenet_zero(weight_pred_file, atten_pred_file):

    test_feat_file_path = []
    testlabels = []
    with open(testlist_folder) as fp:  # test_image_list.txt
        for line in fp:
            fname, lbl = line.split()  # n03236735/n03236735_4047.JPEG 398 [image path, image label]

            assert int(lbl) >= 398
            # images of different dataset have different suffix
            if DATASET == 'ImNet_A':
                feat_name = os.path.join(test_img_feat_folder, fname.replace('.JPEG', '.npz'))  # get the image features
            if DATASET == 'AwA':
                feat_name = os.path.join(test_img_feat_folder, fname.replace('.jpg', '.npz'))  # get the image features

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
    with open(weight_pred_file, 'rb') as fp:  #
        weight_pred = pkl.load(fp)
    weight_pred = np.array(weight_pred)
    # predicted attention weights
    with open(atten_pred_file, 'rb') as fp:  #
        atten_pred = pkl.load(fp)
    atten_pred = np.array(atten_pred)

    print('weight_pred output', weight_pred.shape)

    with open(inv_wordn_file) as fp:
        WNIDS_IN_GRAPH_INDEX = json.load(fp)  # wnid index in graph

    invalid_wv = 0  # count the number of invalid class embedding
    labels_testval, word2vec_testval = [], []  # zsl: unseen label and its class embedding
    weight_pred_testval = []  # zsl: unseen classifier
    atten_pred_testval = []  # zsl: unseen class's attention weights
    wnid_testval = []   # zsl: unseen class's wnid

    for j in range(len(classids)):
        t_wpt = weight_pred[j]
        t_apt = atten_pred[j]
        t_wnid = WNIDS_IN_GRAPH_INDEX[j]

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
            atten_pred_testval.append(t_apt)
            weight_pred_testval.append(t_wpt)
            wnid_testval.append(t_wnid)
    weight_pred_testval = np.array(weight_pred_testval)
    atten_pred_testval = np.array(atten_pred_testval)
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


    labels_testval = np.array(labels_testval)
    print('final test classes: ', len(labels_testval))



    # show the each unseen class's weight attentions
    for i in range(len(wnid_testval)):

        print("-------", i+1, "---", wnid_testval[i], " attention weights:", )

        for h in range(len(atten_pred_testval[i])):
            if atten_pred_testval[i][h] > 1e-2:
                print("- IMSC -", WNIDS_IN_GRAPH_INDEX[h], ":", atten_pred_testval[i][h])







if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='../data', help='root directory')
    parser.add_argument('--data_dir', type=str, default='AZSL-G', help='data directory')
    parser.add_argument('--dataset', type=str, default='ImNet_A', help='ImNet_A, AwA')
    parser.add_argument('--feat', type=str, default='', help='the predicted file')

    args = parser.parse_args()

    DATA_DIR = os.path.join(args.data_root, args.data_dir)
    DATASET = args.dataset

    EXP_NAME = 'Exp2_AGCN'

    feat_dim = 2048  # the dimension of CNN features
    # the word embedidngs of graph nodes, to ensure that the testing classes have initialization
    word2vec_file = os.path.join(DATA_DIR, DATASET, 'glove_w2v.pkl')
    # mark the type of graph nodes (seen, unseen, others)
    classids_file_retrain = os.path.join(DATA_DIR, DATASET, 'corresp.json')
    inv_wordn_file = os.path.join(DATA_DIR, DATASET, 'invdict_wordn.json')  # wnids in graph index


    # testing image list
    # testlist_folder = os.path.join(DATA_DIR, DATASET, 'test_img_list_100.txt')
    testlist_folder = os.path.join(DATA_DIR, DATASET, 'test_img_list.txt')

    # img features of test images
    test_img_feat_folder = os.path.join(DATA_DIR, DATASET, 'Test_DATA_feats')


    # ImNet_A: python test_gcn.py --feat 1500
    # AwA: python test_gcn.py --dataset AwA --feat
    training_outputs = os.path.join(DATA_DIR, DATASET, EXP_NAME, 'feat_' + args.feat)
    # the trained attention weights
    training_coefs = os.path.join(DATA_DIR, DATASET, EXP_NAME, 'coef_' + args.feat)


    print('\nEvaluating ...\nPlease be patient for it takes a few minutes...')

    test_imagenet_zero(weight_pred_file=training_outputs, atten_pred_file=training_coefs)





