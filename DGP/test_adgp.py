import argparse
import json
import os
import os.path as osp
import sys
import time
import torch
from torch.utils.data import DataLoader



from model.utils import pick_vectors




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='/home/gyx/X_ZSL/data', help='root directory')
    parser.add_argument('--data_dir', type=str, default='DGP', help='data directory')
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
    # awa:
    # ImNet: 680
    pred_file = os.path.join(DATA_DIR, DATASET, EXP_NAME, 'epoch-' + args.pred + '.pred')




    # set_gpu(args.gpu)
    '''
    split.json:
    split[train], split[test]
    '''

    split = json.load(open(data_split, 'r'))
    train_wnids = split['train']
    test_wnids = split['test']


    print('train: {}, test: {}'.format(len(train_wnids), len(test_wnids)))
    print('consider train classifiers: {}'.format(args.consider_trains))


    preds = torch.load(pred_file)
    pred_wnids = preds['wnids']
    pred_vectors = preds['pred']  # (3969, 2049)

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


    # total_hits, total_imgs = 0, 0
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
        cls_path = osp.join(test_feat, wnid)
        paths = os.listdir(cls_path)
        feat_data = list()
        for path in paths:
            feat = torch.load(osp.join(cls_path, path))
            feat = torch.squeeze(feat)

            feat_data.append(feat)

        # test 100
        # if len(feat_data) > 100:
        #     feat_data = feat_data[:100]

        feat_data = torch.stack(feat_data, dim=0)
        # print 'feat_data shape:', feat_data.shape
        # load test features end


        feat = torch.cat([feat_data, torch.ones(len(feat_data)).view(-1, 1)], dim=1)

        fcs = pred_vectors.t()  # [2049, 50]

        table = torch.matmul(feat, fcs)
        # False: filter seen classifiers
        if not args.consider_trains:
            table[:, :n] = -1e18

        # for hit@1 and hit@2
        gth_score = table[:, all_label].repeat(table.shape[1], 1).t()
        rks = (table >= gth_score).sum(dim=1)
        assert (table[:, all_label] == gth_score[:, all_label]).min() == 1
        for j, k in enumerate(top):
            hits[j] += (rks <= k).sum().item()
        tot += len(feat_data)

        total_hits += hits
        total_imgs += tot

        # print('{}/{}, {}, total: {} : '.format(i, len(test_wnids), wnid, tot))
        # # hits = float(hits) / float(tot)
        # hits = [float(hit) / float(tot) for hit in hits]
        # output = ['{:.2f}'.format(i * 100) for i in hits]
        # print('results: ', output)

    print('total images: ', total_imgs)
    # total_hits = float(total_hits) / float(total_imgs)
    total_hits = [float(hit) / float(total_imgs) for hit in total_hits]
    output = ['{:.2f}'.format(i * 100) for i in total_hits]
    print('results: ', output)

