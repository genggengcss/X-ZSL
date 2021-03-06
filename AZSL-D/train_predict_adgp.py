import argparse
import json
import random
import os
import sys

import torch
import torch.nn.functional as F



from model import agcn
from model import utils
import val_adgp


'''
input: induced-graph.json, fc-weights.json (training supervision)
output: predicted classifiers
function: train with gcn(2 layers) and predict unseen classifiers
additional attention layer
'''





def save_checkpoint(name):
    torch.save(gcn.state_dict(), os.path.join(save_path, name + '.pth'))
    torch.save(pred_obj, os.path.join(save_path, name + '.pred'))


def mask_l2_loss(a, b, mask):
    return utils.l2_loss(a[mask], b[mask])



def mask_matrix(idx, h, w):
    # 3969*2049
    matrix_zero = torch.zeros(h, w)
    # matrix_one = torch.ones(h, w)
    for index in idx:
        matrix_zero[index, :] = 1

    # mask_one = matrix_zero
    mask = matrix_zero
    mask /= torch.mean(mask)
    mask = mask * mask.t()
    mask = mask * 5
    return mask




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='../data', help='root directory')
    parser.add_argument('--data_dir', type=str, default='AZSL-D', help='data directory')
    parser.add_argument('--dataset', type=str, default='ImNet_A', help='ImNet_A, AwA')
    parser.add_argument('--max_epoch', type=int, default=1500)
    parser.add_argument('--trainval', default='10,0')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--save_epoch', type=int, default=500)
    parser.add_argument('--evaluate_epoch', type=int, default=10)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--trainval', action='store_true', default=False, help='validation set')

    parser.add_argument('--no-pred', action='store_true')
    args = parser.parse_args()


    DATA_DIR = os.path.join(args.data_root, args.data_dir)
    DATASET = args.dataset
    EXP_NAME = 'Exp2_ADGP'


    if DATASET == 'ImNet_A':
        data_split = os.path.join(DATA_DIR, DATASET, 'seen-unseen-split.json')
    if DATASET == 'AwA':
        data_split = os.path.join(DATA_DIR, DATASET, 'awa2-split.json')


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
    print 'fc_vector shape:', fc_vectors.shape
    # add_atten_cos(fc_vectors)



    # seen + unseen mask
    # get seen+unseen index from wnids
    split = json.load(open(data_split, 'r'))
    seen_wnids = train_wnids   # seen
    unseen_wnids = split['test']  # unseen

    seen_index = list()
    unseen_index = list()
    mask_index = list()
    for wnid in seen_wnids:
        mask_index.append(wnids.index(wnid))
        seen_index.append(wnids.index(wnid))
    for wnid in unseen_wnids:
        mask_index.append(wnids.index(wnid))
        unseen_index.append(wnids.index(wnid))

    mask = mask_matrix(mask_index, word_vectors.shape[0], word_vectors.shape[0])  # (3969, 2049)


    # construct gcn model
    hidden_layers = 'd2048,d'
    gcn = agcn.GCN(n, edges, word_vectors.shape[1], fc_vectors.shape[1], hidden_layers, mask)

    print('{} nodes, {} edges'.format(n, len(edges)))
    print('word vectors:', word_vectors.shape)
    print('fc vectors:', fc_vectors.shape)
    print('hidden layers:', hidden_layers)

    optimizer = torch.optim.Adam(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)



    tlist = list(range(len(fc_vectors)))  # 398
    # print "tlist:", tlist
    random.shuffle(tlist)

    min_loss = 1e18

    trlog = {}
    trlog['train_loss'] = []
    trlog['min_loss'] = 0

    # start learning ...
    for epoch in range(1, args.max_epoch + 1):
        # print epoch
        gcn.train()
        output_vectors, coefs = gcn(word_vectors)

        # add attention layer
        # output_vectors_atten, coefs = add_atten_cos(output_vectors)
        # calculate the loss over training seen nodes
        loss = mask_l2_loss(output_vectors, fc_vectors, tlist)
        # print loss
        # loss = mask_l2_loss(output_vectors_atten, fc_vectors, tlist[:n_train])
        optimizer.zero_grad()
        # print "grid?"
        loss.backward()
        # print "loss?"
        optimizer.step()
        # print "optimization?"

        # calculate loss on training (and validation) seen nodes
        if epoch % args.evaluate_epoch == 0:
            gcn.eval()
            output_vectors, coefs = gcn(word_vectors)


            train_loss = mask_l2_loss(output_vectors, fc_vectors, tlist).item()

            loss = train_loss
            # print('epoch {}, train_loss={:.4f}, lr={:.4f}'.format(epoch, train_loss, args.lr))
            print('epoch {}, train_loss={:.4f}'.format(epoch, train_loss))

            trlog['train_loss'].append(train_loss)
            trlog['min_loss'] = min_loss
            torch.save(trlog, os.path.join(save_path, 'trlog'))

        # save intermediate output_vector of each node of the graph
        if epoch >= args.save_epoch and epoch % 10 == 0:
            if args.no_pred:
                pred_obj = None
            else:
                pred_obj = {
                    'wnids': wnids,
                    'pred': output_vectors,
                    'coef': coefs
                }
        # if epoch % args.save_epoch == 0:
        if epoch >= args.save_epoch and epoch % 10 == 0:
            save_checkpoint('epoch-{}'.format(epoch))

        pred_obj = None

        # validation
        if args.trainval:
            if epoch >= args.save_epoch and epoch % 10 == 0:
                if args.no_pred:
                    pred_obj = None
                else:
                    pred_obj = {
                        'wnids': wnids,
                        'pred': output_vectors
                    }
                val_adgp.val(pred_obj, dir=DATA_DIR, dataset=DATASET)
            pred_obj = None


