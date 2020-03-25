import argparse
import json
import random
import os
import sys

import torch
import torch.nn.functional as F




from model import agcn
from model import utils


import test_in_train
'''
input: imagenet-induced-graph.json, fc-weights.json
get: save prediction model file
function: train with gcn(2 layers) and predict testing features
additional attention layer
'''





Mat_DATA_DIR = '/home/gyx/X_ZSL/data/materials'
DATA_DIR = '/home/gyx/X_ZSL/data/DGP'
# DATASET = 'ImNet_A'
DATASET = 'AwA'
EXP_NAME = 'Exp2_ADGP'


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


    return mask



def mask_matrix1(idx, seen_idx, unseen_idx, h, w):
    # 3969*2049
    matrix_all = torch.zeros(h, w)
    matrix_eye = torch.eye(h)

    for a_idx in idx:
        if a_idx in seen_idx:
            for u_idx in unseen_idx:
                matrix_all[a_idx, u_idx] = 1
        if a_idx in unseen_idx:
            for s_idx in seen_idx:
                matrix_all[a_idx, s_idx] = 1


    mask = matrix_all+matrix_eye
    mask[mask > 0.0] = 1.0
    return mask

def add_atten_cos(inputs):

    output = inputs
    print output

    # consin distance
    output = F.normalize(output)
    output_T = output.t()
    logits = torch.mm(output, output_T)

    # coefs = F.softmax(logits, dim=1)
    coefs = logits
    # print coefs.shape
    print coefs

def add_atten_dot(inputs):

    output = inputs
    print output

    # consin distance
    # output = F.normalize(output)
    output_T = output.t()
    logits = torch.mm(output, output_T)

    # coefs = F.softmax(logits, dim=1)
    coefs = logits
    # print coefs.shape
    print coefs

# save_epochs = [10, 300, 350, 400, 450, 500]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--graph',
                        default=os.path.join(DATA_DIR, 'materials', 'induced-graph.json'))
    parser.add_argument('--max_epoch', type=int, default=3000)
    parser.add_argument('--trainval', default='10,0')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--save_epoch', type=int, default=800)
    parser.add_argument('--save_path', default=os.path.join(DATA_DIR, DATASET, EXP_NAME))
    parser.add_argument('--evaluate_epoch', type=int, default=10)
    parser.add_argument('--awa2_split', default=os.path.join(DATA_DIR, DATASET, 'awa2-split.json'))

    parser.add_argument('--gpu', default='1')

    parser.add_argument('--no-pred', action='store_true')
    args = parser.parse_args()

    # needed?
    # set_gpu(args.gpu)

    save_path = args.save_path
    utils.ensure_path(save_path)

    graph = json.load(open(args.graph, 'r'))
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



    # seen + unseen mask begin
    # get seen+unseen index from wnids
    awa2_split = json.load(open(args.awa2_split, 'r'))
    seen_wnids = train_wnids   # seen
    unseen_wnids = awa2_split['test']  # unseen

    seen_index = list()
    unseen_index = list()
    mask_index = list()
    for wnid in seen_wnids:
        mask_index.append(wnids.index(wnid))
        seen_index.append(wnids.index(wnid))
    for wnid in unseen_wnids:
        mask_index.append(wnids.index(wnid))
        unseen_index.append(wnids.index(wnid))
    # print "seen index:", seen_index
    # print "unseen index:", unseen_index
    # print 'mask index:', mask_index
    # seen + unseen mask end
    mask = mask_matrix(mask_index, word_vectors.shape[0], word_vectors.shape[0])  # (3969, 2049)

    # mask = mask_matrix1(mask_index, seen_index, unseen_index, word_vectors.shape[0], word_vectors.shape[0])  # (3969, 3969)

    # construct gcn model
    hidden_layers = 'd2048,d'
    gcn = agcn.GCN(n, edges, word_vectors.shape[1], fc_vectors.shape[1], hidden_layers, mask)

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
    # print "tlist:", tlist
    random.shuffle(tlist)

    min_loss = 1e18

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['min_loss'] = 0

    # start learning ...
    for epoch in range(1, args.max_epoch + 1):
        # print epoch
        gcn.train()
        output_vectors, coefs = gcn(word_vectors)

        # add attention layer
        # output_vectors_atten, coefs = add_atten_cos(output_vectors)
        # calculate the loss over training seen nodes
        loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train])
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


            train_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train]).item()
            # if v_val > 0:
            #     val_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[n_train:]).item()
            #     loss = val_loss
            # else:
            #     val_loss = 0
            #     loss = train_loss
            print('epoch {}, train_loss={:.4f}, lr={:.4f}'.format(epoch, train_loss, args.lr))

            trlog['train_loss'].append(train_loss)
            # trlog['val_loss'].append(val_loss)
            trlog['min_loss'] = min_loss
            torch.save(trlog, os.path.join(save_path, 'trlog'))

        # # save intermediate output_vector of each node of the graph
        # if epoch >= args.save_epochs and epoch % 100 == 0:
        #     if args.no_pred:
        #         pred_obj = None
        #     else:
        #         pred_obj = {
        #             'wnids': wnids,
        #             'pred': output_vectors
        #         }
        # if epoch >= args.save_epochs and epoch % 100 == 0:
        #
        #     save_checkpoint('epoch-{}'.format(epoch))
        #
        # pred_obj = None





        # # if epoch == 10 or (epoch >= 100 and epoch % 10 == 0):
        if epoch == 10 or (epoch >= 800 and epoch % 10 == 0):
            if args.no_pred:
                pred_obj = None
            else:
                pred_obj = {
                    'wnids': wnids,
                    'pred': output_vectors
                }
            test_in_train.test_in_train(pred_obj)
        pred_obj = None


        # # calculate loss on training (and validation) seen nodes
        # if epoch % args.evaluate_epoch == 0:
        #     gcn.eval()
        #     output_vectors = gcn(word_vectors)
        #     train_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train]).item()
        #     if v_val > 0:
        #         val_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[n_train:]).item()
        #         loss = val_loss
        #     else:
        #         val_loss = 0
        #         loss = train_loss
        #     print('epoch {}, train_loss={:.4f}, val_loss={:.4f}'.format(epoch, train_loss, val_loss))
        #
        #     trlog['train_loss'].append(train_loss)
        #     trlog['val_loss'].append(val_loss)
        #     trlog['min_loss'] = min_loss
        #     torch.save(trlog, os.path.join(save_path, 'trlog'))
        #
        # # save intermediate output_vector of each node of the graph
        # if epoch % args.save_epoch == 0:
        #     if args.no_pred:
        #         pred_obj = None
        #     else:
        #         pred_obj = {
        #             'wnids': wnids,
        #             'pred': output_vectors
        #         }
        # if epoch % args.save_epoch == 0:
        #     save_checkpoint('epoch-{}'.format(epoch))
        #
        # pred_obj = None

