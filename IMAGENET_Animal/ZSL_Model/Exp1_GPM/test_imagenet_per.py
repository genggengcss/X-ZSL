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

wnid_wname_file = os.path.join(DATA_DIR_PREFIX, 'materials', 'words_animal.txt')

def obtain_wnid_wname_dict():
    words = dict()
    with open(wnid_wname_file, 'r') as fp:
        for line in fp.readlines():
            wnid, name = line.split('\t')
            wnid = wnid.strip()
            name = name.strip()
            words[wnid] = name
    return words

WNID_WNAME = obtain_wnid_wname_dict()  # wnid and its word name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', default=os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'save-gpm1/epoch-400.pred'))
    parser.add_argument('--test_feat', default=os.path.join(DATA_DIR_PREFIX, 'IMAGENET_Animal/Test_DATA_feats_GPM'))
    parser.add_argument('--imagenet_split', default=os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'imagenet-train-test-split.json'))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--consider-trains', action='store_true')

    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    # set_gpu(args.gpu)
    '''
    imagenet-train-test-split.json:
    split[train], split[test]
    '''

    imagenet_split = json.load(open(args.imagenet_split, 'r'))
    train_wnids = imagenet_split['train']
    test_wnids = imagenet_split['test']


    print('train: {}, test: {}'.format(len(train_wnids), len(test_wnids)))
    print('consider train classifiers: {}'.format(args.consider_trains))


    pred_file = torch.load(args.pred)
    pred_wnids = pred_file['wnids']
    pred_vectors = pred_file['pred']  # (3969, 2049)

    pred_dic = dict(zip(pred_wnids, pred_vectors))  # packed into tuple
    # select seen and unseen pred_vectors
    pred_vectors = utils.pick_vectors(pred_dic, train_wnids + test_wnids, is_tensor=True)

    n = len(train_wnids)
    m = len(test_wnids)


    # test_names = awa2_split['test_names']

    ave_acc = 0
    ave_acc_n = 0

    results = {}

    # the directory of AWA2 testing data features
    imagenet_test_path = args.test_feat

    # total_hits, total_imgs = 0, 0
    total_hits = torch.FloatTensor([0, 0, 0, 0, 0])  # top 1 2 5 10 20
    total_imgs = 0
    per_dict = dict()
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

        wname = WNID_WNAME[wnid]
        per_dict[wname] = list()
        per_dict[wname].append([float(hit) for hit in hits])
        per_dict[wname].append(tot)

        print('{}/{}, {}, {}, total: {} : '.format(i, len(test_wnids), wnid, wname, tot))
        # hits = float(hits) / float(tot)
        hits = [float(hit) / float(tot) for hit in hits]
        output = ['{:.2f}'.format(i * 100) for i in hits]
        print('results: ', output)

    # save to file
    per_class_acc = 'gcn_per_class_acc1.txt'
    wr_fp = open(per_class_acc, 'w')
    for name, result in per_dict.items():
        line = name
        for res in result[0]:
            line = line + "\t" + str(res)
        line = line + "\t" + str(result[1])
        # print(line)
        wr_fp.write('%s\n' % line)
    wr_fp.close()



    print('total images: ', total_imgs)
    # total_hits = float(total_hits) / float(total_imgs)
    total_hits = [float(hit) / float(total_imgs) for hit in total_hits]
    output = ['{:.2f}'.format(i * 100) for i in total_hits]
    print('results: ', output)

    if args.output is not None:
        json.dump(results, open(args.output, 'w'))
