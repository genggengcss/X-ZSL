import argparse
import json
import os
import os.path as osp
import sys
import time
import torch
from torch.utils.data import DataLoader


sys.path.append('../../../../')
from ZSL.gpm import resnet
from ZSL.gpm import utils
from ZSL.gpm import image_folder




DATA_DIR_PREFIX = '/home/gyx/Data/KG_SS_DATA/ZSL/'
# DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'
EXP_NAME = 'IMAGENET_Animal/Exp1_GPM'

test_feat = os.path.join(DATA_DIR_PREFIX, 'IMAGENET_Animal/Test_DATA_feats_GPM')



def test_in_train(pred):

    '''
    imagenet-train-test-split.json:
    split[train], split[test]
    '''
    imagenet_split = json.load(open(os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'imagenet-train-test-split.json'),'r'))
    train_wnids = imagenet_split['train']
    test_wnids = imagenet_split['test']


    # print('train: {}, test: {}'.format(len(train_wnids), len(test_wnids)))
    # print('consider train classifiers: {}'.format(args.consider_trains))


    pred_file = pred
    pred_wnids = pred_file['wnids']
    pred_vectors = pred_file['pred']  # (3969, 2049)

    pred_dic = dict(zip(pred_wnids, pred_vectors))  # packed into tuple
    # select seen and unseen pred_vectors
    pred_vectors = utils.pick_vectors(pred_dic, train_wnids + test_wnids, is_tensor=True)

    n = len(train_wnids)
    m = len(test_wnids)


    # test_names = awa2_split['test_names']



    # the directory of Imagenet testing data features
    imagenet_test_path = test_feat

    total_hits = torch.FloatTensor([0, 0, 0, 0, 0])  # top 1 2 5 10 20
    total_imgs = 0
    for i, wnid in enumerate(test_wnids, 1):
        all_label = n + i - 1
        # hit = 0
        # tot = 0
        top = [1, 2, 5, 10, 20]
        hits = torch.zeros(len(top))
        tot = 0


        # load test features begin
        cls_path = osp.join(imagenet_test_path, wnid)
        paths = os.listdir(cls_path)
        if len(paths) <= 100:
            paths = paths
        else:
            paths = paths[0:99]

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

        # for accuracy
        # pred = torch.argmax(table, dim=1)
        # hit += (pred == all_label).sum().item()
        # tot += len(feat_data)

        # for hit@1 and hit@2
        gth_score = table[:, all_label].repeat(table.shape[1], 1).t()
        rks = (table >= gth_score).sum(dim=1)
        assert (table[:, all_label] == gth_score[:, all_label]).min() == 1
        for j, k in enumerate(top):
            hits[j] += (rks <= k).sum().item()
        tot += len(feat_data)


        total_hits += hits
        total_imgs += tot

        # print('{}/{}, {}, total:{}: '.format(i, len(test_wnids), wnid, tot))
        # hits = float(hits) / float(tot)

        hits = [float(hit)/float(tot) for hit in hits]
        output = ['{:.2f}'.format(i * 100) for i in hits]
        # print('results: ', output)



    # print('-- Total Images: --', total_imgs)
    # total_hits = float(total_hits) / float(total_imgs)
    total_hits = [float(hit) / float(total_imgs) for hit in total_hits]
    output = ['{:.2f}'.format(i * 100) for i in total_hits]
    print('-- Total Results: --', output)

