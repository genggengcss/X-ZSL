import argparse
import json
import os
import sys

import torch
# from torch.utils.data import DataLoader

sys.path.append('../../../../')
from ZSL.gpm import resnet
# from ZSL.gpm import imagenet
from ZSL.gpm import utils



# DATA_DIR_PREFIX = '/home/gyx/Data/KG_SS_DATA/ZSL/'
DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'
EXP_NAME = 'AWA2/Exp1_GPM'

def test_on_subset(dataset, cnn, n, pred_vectors, all_label,
                   consider_trains):
    top = [1, 2, 5, 10, 20]
    hits = torch.zeros(len(top))
    tot = 0

    loader = DataLoader(dataset=dataset, batch_size=32,
                        shuffle=False, num_workers=2)

    for batch_id, batch in enumerate(loader, 1):
        data, label = batch
        # data = data.cuda()

        feat = cnn(data)  # (batch_size, d)
        feat = torch.cat([feat, torch.ones(len(feat)).view(-1, 1)], dim=1)

        fcs = pred_vectors.t()

        table = torch.matmul(feat, fcs)
        if not consider_trains:
            table[:, :n] = -1e18

        gth_score = table[:, all_label].repeat(table.shape[1], 1).t()
        rks = (table >= gth_score).sum(dim=1)

        assert (table[:, all_label] == gth_score[:, all_label]).min() == 1

        for i, k in enumerate(top):
            hits[i] += (rks <= k).sum().item()
        tot += len(data)

    return hits, tot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn', default=os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'resnet50-base.pth'))
    parser.add_argument('--pred')

    parser.add_argument('--test-set', default='2-hops')

    parser.add_argument('--output', default=None)

    parser.add_argument('--gpu', default='0')

    parser.add_argument('--keep-ratio', type=float, default=0.1)
    parser.add_argument('--consider-trains', action='store_true')
    parser.add_argument('--test-trains', action='store_true')

    args = parser.parse_args()

    # set_gpu(args.gpu)

    test_sets = json.load(open(os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'imagenet-testsets.json'), 'r'))

    train_wnids = test_sets['train']  # seen classes

    test_wnids = test_sets[args.test_set]  # 2-hops, unseen classes

    print('test set: {}, {} classes, ratio={}'.format(args.test_set, len(test_wnids), args.keep_ratio))
    print('consider train classifiers: {}'.format(args.consider_trains))

    pred_file = torch.load(args.pred)
    pred_wnids = pred_file['wnids']
    pred_vectors = pred_file['pred']
    pred_dic = dict(zip(pred_wnids, pred_vectors))
    pred_vectors = utils.pick_vectors(pred_dic, train_wnids + test_wnids, is_tensor=True)

    n = len(train_wnids)
    m = len(test_wnids)

    cnn = resnet.make_resnet50_base()
    cnn.load_state_dict(torch.load(args.cnn))
    cnn.eval()

    TEST_TRAIN = args.test_trains

    imagenet_path = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'imagenet')
    dataset = imagenet.ImageNet(imagenet_path)
    dataset.set_keep_ratio(args.keep_ratio)

    s_hits = torch.FloatTensor([0, 0, 0, 0, 0])  # top 1 2 5 10 20
    s_tot = 0

    results = {}

    if TEST_TRAIN:
        for i, wnid in enumerate(train_wnids, 1):
            subset = dataset.get_subset(wnid)
            hits, tot = test_on_subset(subset, cnn, n, pred_vectors, i - 1,
                                       consider_trains=args.consider_trains)
            results[wnid] = (hits / tot).tolist()

            s_hits += hits
            s_tot += tot

            print('{}/{}, {}:'.format(i, len(train_wnids), wnid))
            for j in range(len(hits)):
                print('{:.0f}%({:.2f}%)'.format(hits[j] / tot * 100, s_hits[j] / s_tot * 100))
            print('x{}({})'.format(tot, s_tot))
    else:
        for i, wnid in enumerate(test_wnids, 1):
            subset = dataset.get_subset(wnid)
            hits, tot = test_on_subset(subset, cnn, n, pred_vectors, n + i - 1,
                                       consider_trains=args.consider_trains)
            results[wnid] = (hits / tot).tolist()

            s_hits += hits
            s_tot += tot

            print('{}/{}, {}:'.format(i, len(test_wnids), wnid))
            for j in range(len(hits)):
                print('{:.0f}%({:.2f}%)'.format(hits[j] / tot * 100, s_hits[j] / s_tot * 100))
            print('x{}({})'.format(tot, s_tot))

    print('summary:')
    for s_hit in s_hits:
        print('{:.2f}%'.format(s_hit / s_tot * 100))
    print('total {}'.format(s_tot))

    if args.output is not None:
        json.dump(results, open(args.output, 'w'))

