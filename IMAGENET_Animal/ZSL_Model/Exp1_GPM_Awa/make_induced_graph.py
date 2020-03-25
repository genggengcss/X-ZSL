import argparse
import json
import os
import sys
import torch

from nltk.corpus import wordnet as wn

sys.path.append('../../../../')
from ZSL.gpm import myglove

'''
input: imagenet-split.json, imagenet-xml-wnids.json
get:  'imagenet-induced-graph.json'
function: construct graph (total nodes) and get node embedding
'''

# DATA_DIR_PREFIX = '/home/gyx/Data/KG_SS_DATA/ZSL/'
DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'
EXP_NAME = 'IMAGENET_Animal/Exp1_GPM'


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
    parser.add_argument('--input', default=os.path.join(DATA_DIR_PREFIX, 'IMAGENET_Animal/materials', 'imagenet-animal-split.json'))
    parser.add_argument('--output', default=os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'imagenet-induced-animal-graph.json'))
    args = parser.parse_args()

    print('making graph ...')
    # imagenet animal subset, length: 32295
    xml_wnids = json.load(open(os.path.join(DATA_DIR_PREFIX, 'IMAGENET_Animal/materials', 'imagenet-xml-animal-wnids.json'), 'r'))
    print 'xml_nodes:', len(xml_wnids)
    xml_nodes = list(map(getnode, xml_wnids))
    xml_set = set(xml_nodes)  # get wordnet node text

    # imagenet animal subset split
    js = json.load(open(args.input, 'r'))
    # print js
    train_wnids = js['train']  # 1000
    test_wnids = js['test']  # 20842
    print 'train_wnids:', len(train_wnids)
    print 'test_wnids:', len(test_wnids)

    key_wnids = train_wnids + test_wnids

    s = list(map(getnode, key_wnids))  # get train+test node text

    '''Actually this does not make any changes, as all parents of nodes in s are among xml_set
        s: 21842
    '''
    induce_parents(s, xml_set)

    # len(s) : 21842
    # len(s_set) : 21842
    # len(s) : 32324  p.s. new num of s
    s_set = set(s)
    for u in xml_nodes:
        if u not in s_set:
            s.append(u)
    '''
        new s: 32324, xml_nodes: 32295
        this means 29 nodes of train+test are not among xml_nodes

        the function of above step: construct complete graph (or node set), the total number is: 32324  
    '''
    print len(s)

    wnids = list(map(getwnid, s))  # get s's wnids (graph nodes)
    print wnids
    edges = getedges(s)

    print('making glove embedding ...')

    glove = myglove.GloVe(os.path.join(DATA_DIR_PREFIX, 'materials', 'glove.6B.300d.txt'))
    vectors = []
    for wnid in wnids:
        vectors.append(glove[getnode(wnid).lemma_names()])
    vectors = torch.stack(vectors)

    print('dumping ...')

    obj = {'wnids': wnids, 'vectors': vectors.tolist(), 'edges': edges}
    json.dump(obj, open(args.output, 'w'))
