import argparse
import json
import os
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
EXP_NAME = 'AWA2/Exp1_GPM'


def test_on_subset(dataset, cnn, n, pred_vectors, all_label, consider_trains):
    hit = 0
    tot = 0

    loader = DataLoader(dataset=dataset, batch_size=32,
                        shuffle=False, num_workers=2)

    for batch_id, batch in enumerate(loader, 1):
        # print "batch_id: ", batch_id, " batch:", batch
        data, label = batch
        # print "data: ", data, "label: ", label
        print "data shape: ", data.shape, "label shape: ", label.shape
        feat = cnn(data)  # (batch_size, d)
        feat = torch.cat([feat, torch.ones(len(feat)).view(-1, 1)], dim=1)

        fcs = pred_vectors.t()

        table = torch.matmul(feat, fcs)
        if not consider_trains:
            table[:, :n] = -1e18

        pred = torch.argmax(table, dim=1)
        hit += (pred == all_label).sum().item()
        tot += len(data)

    return hit, tot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn', default=os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'resnet50-base.pth'))
    parser.add_argument('--pred', default=os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'save-gpm/epoch-300.pred'))

    parser.add_argument('--gpu', default='0')
    parser.add_argument('--consider-trains', action='store_true')

    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    # set_gpu(args.gpu)
    '''
    awa2-split.json:
    awa2_split[train], awa2_split[test], awa2_split[train_names], awa2_split[test_names]
    '''

    awa2_split = json.load(open(os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'awa2-split.json'), 'r'))
    train_wnids = awa2_split['train']
    test_wnids = awa2_split['test']
    # print train_wnids
    # print test_wnids

    print('train: {}, test: {}'.format(len(train_wnids), len(test_wnids)))
    print('consider train classifiers: {}'.format(args.consider_trains))


    pred_file = torch.load(args.pred)
    pred_wnids = pred_file['wnids']
    pred_vectors = pred_file['pred']
    pred_dic = dict(zip(pred_wnids, pred_vectors))  # packed into tuple
    # select seen and unseen pred_vectors
    pred_vectors = utils.pick_vectors(pred_dic, train_wnids + test_wnids, is_tensor=True)

    n = len(train_wnids)
    m = len(test_wnids)

    cnn = resnet.make_resnet50_base()
    cnn.load_state_dict(torch.load(args.cnn))
    cnn.eval()

    test_names = awa2_split['test_names']

    ave_acc = 0
    ave_acc_n = 0

    results = {}

    # the directory of raw AWA2 data, with images
    awa2_path = os.path.join('/home/gyx/Data/Ori_DATA', 'AwA2', 'JPEGImages')

    total_hits, total_imgs = 0, 0
    for i, name in enumerate(test_names, 1):
        t = time.time()
        print i, ' ', name
        dataset = image_folder.ImageFolder(awa2_path, [name], 'test')
        print 'dataset: ', dataset
        print("time1=", "{:.5f}".format(time.time() - t))
        t = time.time()
        hit, tot = test_on_subset(dataset, cnn, n, pred_vectors, n + i - 1,
                                  args.consider_trains)
        print("time2=", "{:.5f}".format(time.time() - t))
        total_hits += hit
        total_imgs += tot

        acc = float(hit) / float(tot)
        ave_acc += acc
        ave_acc_n += 1

        print('{} {}: {:.2f}%'.format(i, name.replace('+', ' '), acc * 100))
        print('hit: {}, tot: {}'.format(hit, tot))

        results[name] = acc

    print('\nper-class accuracy: {:.2f}%'.format(ave_acc / ave_acc_n * 100))

    overall_acc = float(total_hits) / float(total_imgs)
    print('\noverall accuracy: {:.2f}%'.format(overall_acc * 100))

    if args.output is not None:
        json.dump(results, open(args.output, 'w'))
