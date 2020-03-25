import argparse
import json
import os
import os.path as osp
import sys
import time
import torch
from torch.utils.data import DataLoader




from model.utils import pick_vectors




Mat_DATA_DIR = '/home/gyx/X_ZSL/data/materials'
DATA_DIR = '/home/gyx/X_ZSL/data/DGP'
# DATASET = 'ImNet_A'
DATASET = 'AwA'

test_feat = os.path.join(DATA_DIR, DATASET, 'Test_DATA_feats')


def test_in_train(pred):

    '''
    awa2-split.json:
    awa2_split[train], awa2_split[test], awa2_split[train_names], awa2_split[test_names]
    '''
    awa2_split = json.load(open(os.path.join(DATA_DIR, DATASET, 'awa2-split.json'), 'r'))
    train_wnids = awa2_split['train']
    test_wnids = awa2_split['test']


    # print('train: {}, test: {}'.format(len(train_wnids), len(test_wnids)))
    # print('consider train classifiers: {}'.format(args.consider_trains))


    pred_file = pred
    pred_wnids = pred_file['wnids']
    pred_vectors = pred_file['pred']  # (3969, 2049)

    pred_dic = dict(zip(pred_wnids, pred_vectors))  # packed into tuple
    # select seen and unseen pred_vectors
    pred_vectors = pick_vectors(pred_dic, train_wnids + test_wnids, is_tensor=True)

    n = len(train_wnids)
    m = len(test_wnids)


    # test_names = awa2_split['test_names']

    ave_acc = 0
    ave_acc_n = 0

    results = {}

    # the directory of AWA2 testing data features
    awa2_test_path = test_feat

    total_hits, total_imgs = 0, 0
    for i, name in enumerate(test_wnids, 1):
        all_label = n + i - 1
        hit = 0
        tot = 0


        # load test features begin
        cls_path = osp.join(awa2_test_path, name)
        paths = os.listdir(cls_path)
        feat_data = list()
        for path in paths:
            feat = torch.load(osp.join(cls_path, path))
            feat = torch.squeeze(feat)

            feat_data.append(feat)

        feat_data = torch.stack(feat_data, dim=0)
        # print 'feat_data shape:', feat_data.shape
        # load test features end


        feat = torch.cat([feat_data, torch.ones(len(feat_data)).view(-1, 1)], dim=1)

        fcs = pred_vectors.t()  # [2049, 50]

        table = torch.matmul(feat, fcs)
        # if not args.co nsider_trains:
        table[:, :n] = -1e18  # filter seen classifiers

        pred = torch.argmax(table, dim=1)
        hit += (pred == all_label).sum().item()
        tot += len(feat_data)


        total_hits += hit
        total_imgs += tot

        acc = float(hit) / float(tot)
        ave_acc += acc
        ave_acc_n += 1

        # print('{} {}: {:.2f}%'.format(i, name.replace('+', ' '), acc * 100))
        # print('hit: {}, tot: {}'.format(hit, tot))

        results[name] = acc

    # print('\nper-class accuracy: {:.2f}%'.format(ave_acc / ave_acc_n * 100))

    overall_acc = float(total_hits) / float(total_imgs)
    print('overall accuracy: {:.2f}%'.format(overall_acc * 100))


# if __name__ == '__main__':
#     pred_path = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'save-gpm1/epoch-300.pred')
#     pred = torch.load(pred_path)
#     test_in_train(pred)