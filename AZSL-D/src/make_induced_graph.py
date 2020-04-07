import argparse
import json
import os
import sys
import torch

from nltk.corpus import wordnet as wn

sys.path.append('../')
from model import myglove

'''
input: imagenet-split-animal.json, imagenet-wnids-animal.json (animal subset)
output:  'induced-graph.json'
function: construct graph (total nodes) and get node embedding
'''



def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))


def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s


# check every node's parent (not including ancestors > 2-hops)
def getedges(s):
    dic = {x: i for i, x in enumerate(s)}
    edges = []
    for i, u in enumerate(s):
        for v in u.hypernyms():
            j = dic.get(v)
            if j is not None:
                edges.append((i, j))
    return edges


# add the parents of nodes of s that are not among stop_set to s
def induce_parents(s, stop_set):
    q = s
    vis = set(s)
    l = 0
    while l < len(q):
        u = q[l]
        l += 1
        if u in stop_set:
            continue
        for p in u.hypernyms():
            if p not in vis:
                vis.add(p)
                q.append(p)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='../../data', help='root directory')
    parser.add_argument('--data_dir', type=str, default='AZSL-D', help='data directory')

    args = parser.parse_args()

    DATA_DIR = os.path.join(args.data_root, args.data_dir)

    input_file = os.path.join(DATA_DIR, 'materials', 'imagenet-split-animal.json')
    output_file = os.path.join(DATA_DIR, 'materials', 'induced-graph.json')

    print('making graph ...')
    # wnids of imagenet animal subset, total: 3969
    xml_wnids = json.load(open(os.path.join(DATA_DIR, 'materials', 'imagenet-wnids-animal.json'), 'r'))
    print 'xml_nodes:', len(xml_wnids)
    xml_nodes = list(map(getnode, xml_wnids))
    xml_set = set(xml_nodes)  # get wordnet node text

    # data split (seen: 398; other: 3395)
    js = json.load(open(input_file, 'r'))
    # print js
    train_wnids = js['train']  # 398
    test_wnids = js['test']  # 3395
    print 'train_wnids:', len(train_wnids)
    print 'test_wnids:', len(test_wnids)

    key_wnids = train_wnids + test_wnids

    s = list(map(getnode, key_wnids))  # get train+test node text

    '''Actually this does not make any changes, as all parents of nodes in s are among xml_set
        s: 21842
    '''
    induce_parents(s, xml_set)

    # len(s) : 3969
    # len(s_set) : 3969
    # len(s) : 3969
    s_set = set(s)
    for u in xml_nodes:
        if u not in s_set:
            s.append(u)

    print len(s)

    wnids = list(map(getwnid, s))  # get s's wnids (graph nodes)
    edges = getedges(s)

    print('making glove embedding ...')

    glove = myglove.GloVe(os.path.join(args.data_root, 'glove.6B.300d.txt'))
    vectors = []
    for wnid in wnids:
        vectors.append(glove[getnode(wnid).lemma_names()])
    vectors = torch.stack(vectors)

    print('dumping ...')

    obj = {'wnids': wnids, 'vectors': vectors.tolist(), 'edges': edges}
    json.dump(obj, open(output_file, 'w'))
