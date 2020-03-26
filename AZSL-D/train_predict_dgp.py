import argparse
import json
import random
import os
import sys

import torch
import torch.nn.functional as F



from model import gcn
from model import utils


'''
input: imagenet-induced-animal-graph.json, fc-weights.json
get: save prediction model file
function: train with gcn(2 layers) and predict testing features
'''




def save_checkpoint(name):
    torch.save(gcn.state_dict(), os.path.join(save_path, name + '.pth'))
    torch.save(pred_obj, os.path.join(save_path, name + '.pred'))


def mask_l2_loss(a, b, mask):
    return utils.l2_loss(a[mask], b[mask])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='/home/gyx/X-ZSL/data', help='root directory')
    parser.add_argument('--data_dir', type=str, default='AZSL-D', help='data directory')
    parser.add_argument('--dataset', type=str, default='ImNet_A', help='ImNet_A, AwA')
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--trainval', default='10,0')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--save_epoch', type=int, default=250)
    parser.add_argument('--evaluate_epoch', type=int, default=10)

    parser.add_argument('--gpu', default='0')

    parser.add_argument('--no-pred', action='store_true')

    args = parser.parse_args()

    DATA_DIR = os.path.join(args.data_root, args.data_dir)
    DATASET = args.dataset
    EXP_NAME = 'Exp1_DGP'

    graph_file = os.path.join(DATA_DIR, 'materials', 'induced-graph.json')
    save_path = os.path.join(DATA_DIR, DATASET, EXP_NAME)

    utils.ensure_path(save_path)

    graph = json.load(open(graph_file, 'r'))
    wnids = graph['wnids']
    n = len(wnids)
    edges = graph['edges']

    # + inverse edges and reflexive edges
    edges = edges + [(v, u) for (u, v) in edges]
    edges = edges + [(u, u) for u in range(n)]

    word_vectors = torch.tensor(graph['vectors'])
    word_vectors = F.normalize(word_vectors)

    # training supervision
    fcfile = json.load(open(os.path.join(DATA_DIR, 'materials', 'fc-weights.json'), 'r'))
    train_wnids = [x[0] for x in fcfile]  # fcfile[0]: wnid list (len:1000)

    assert train_wnids == wnids[:len(train_wnids)]
    fc_vectors = [x[1] for x in fcfile]

    fc_vectors = torch.tensor(fc_vectors)
    fc_vectors = F.normalize(fc_vectors)  # shape: (1000, 2049)


    # construct gcn model
    hidden_layers = 'd2048,d'
    gcn = gcn.GCN(n, edges, word_vectors.shape[1], fc_vectors.shape[1], hidden_layers)

    print('{} nodes, {} edges'.format(n, len(edges)))
    print('word vectors:', word_vectors.shape)
    print('fc vectors:', fc_vectors.shape)
    print('hidden layers:', hidden_layers)

    optimizer = torch.optim.Adam(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # split seen nodes into training set and validation set
    v_train, v_val = map(float, args.trainval.split(','))  # 10, 0
    n_trainval = len(fc_vectors)  # 1000, training number?
    n_train = int(round(n_trainval * (v_train / (v_train + v_val))))
    print('num train: {}, num val: {}'.format(n_train, n_trainval - n_train))  # 1000, 0

    tlist = list(range(len(fc_vectors)))  # 1000
    random.shuffle(tlist)

    min_loss = 1e18

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['min_loss'] = 0

    # start learning ...
    for epoch in range(1, args.max_epoch + 1):
        gcn.train()
        output_vectors = gcn(word_vectors)

        # calculate the loss over training seen nodes
        loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate loss on training (and validation) seen nodes
        if epoch % args.evaluate_epoch == 0:
            gcn.eval()
            output_vectors = gcn(word_vectors)
            train_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train]).item()
            if v_val > 0:
                val_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[n_train:]).item()
                loss = val_loss
            else:
                val_loss = 0
                loss = train_loss
            print('epoch {}, train_loss={:.4f}, val_loss={:.4f}'.format(epoch, train_loss, val_loss))

            trlog['train_loss'].append(train_loss)
            trlog['val_loss'].append(val_loss)
            trlog['min_loss'] = min_loss
            torch.save(trlog, os.path.join(save_path, 'trlog'))

        # save intermediate output_vector of each node of the graph
        if epoch % 50 == 0 and epoch >= args.save_epoch:
            if args.no_pred:
                pred_obj = None
            else:
                pred_obj = {
                    'wnids': wnids,
                    'pred': output_vectors
                }
        if epoch % 50 == 0 and epoch >= args.save_epoch:
            save_checkpoint('epoch-{}'.format(epoch))

        pred_obj = None


        # if epoch == 10 or (epoch % 50 == 0 and epoch >= args.save_epoch):
        #     if args.no_pred:
        #         pred_obj = None
        #     else:
        #         pred_obj = {
        #             'wnids': wnids,
        #             'pred': output_vectors
        #         }
        #     test_in_train.test_in_train(pred_obj)
        # pred_obj = None

