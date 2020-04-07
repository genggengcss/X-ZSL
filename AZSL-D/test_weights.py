'''
extract the attention weights and impressive seen classes from predicted results
'''

import argparse
import json
import os
import torch
import numpy as np

from nltk.corpus import wordnet as wn

from model.utils import pick_vectors



def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='../data', help='root directory')
    parser.add_argument('--data_dir', type=str, default='AZSL-D', help='data directory')
    parser.add_argument('--dataset', type=str, default='ImNet_A', help='ImNet_A, AwA')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--consider-trains', action='store_true')
    parser.add_argument('--no-pred', action='store_true')
    parser.add_argument('--pred', type=str, default='', help='the predicted classifier name')

    args = parser.parse_args()

    DATA_DIR = os.path.join(args.data_root, args.data_dir)
    DATASET = args.dataset
    EXP_NAME = 'Exp2_ADGP'

    # the directory of testing data features
    test_feat = os.path.join(DATA_DIR, DATASET, 'Test_DATA_feats')

    if DATASET == 'ImNet_A':
        data_split = os.path.join(DATA_DIR, DATASET, 'seen-unseen-split.json')
    if DATASET == 'AwA':
        data_split = os.path.join(DATA_DIR, DATASET, 'awa2-split.json')

    # predict classifiers
    # awa: 1200 - 85.52
    # ImNet: 680
    pred_file = os.path.join(DATA_DIR, DATASET, EXP_NAME, 'epoch-' + args.pred + '.pred')





    # split.json: train, test

    split = json.load(open(data_split, 'r'))
    train_wnids = split['train']
    test_wnids = split['test']



    print('train: {}, test: {}'.format(len(train_wnids), len(test_wnids)))
    print('consider train classifiers: {}'.format(args.consider_trains))


    preds = torch.load(pred_file)
    pred_wnids = preds['wnids']
    pred_vectors = preds['pred']  # (3969, 2049)
    pred_coefs = preds['coef']

    pred_dic = dict(zip(pred_wnids, pred_vectors))  # packed into tuple
    coef_dic = dict(zip(pred_wnids, pred_coefs))  # packed into tuple
    # select seen and unseen pred_vectors
    pred_vectors = pick_vectors(pred_dic, train_wnids + test_wnids, is_tensor=True)
    # coef = pick_vectors(coef_dic, train_wnids + test_wnids, is_tensor=True)

    n = len(train_wnids)
    m = len(test_wnids)


    for wnid in test_wnids:

        coefs = coef_dic[wnid]
        print "--------", getnode(wnid).lemma_names()[0], "-------"
        for i, coef in enumerate(coefs):
            if coef != 0 and coef > 1e-2:

                wnid = pred_wnids[i]
                name = getnode(wnid).lemma_names()[0]
                print ("- coef: %.2f, name: %s - " % (coef, name))




