import argparse
import json
import os
import sys
import torch

from nltk.corpus import wordnet as wn


'''
input: imagenet-split.json, imagenet-xml-wnids.json
get:  'imagenet-induced-graph.json'
function: construct graph (total nodes) and get node embedding
'''

# DATA_DIR_PREFIX = '/home/gyx/Data/KG_SS_DATA/ZSL/'
DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'
EXP_NAME = 'AWA2/Exp1_GPM'


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
        print type(u)
        for v in u.hypernyms():
            j = dic.get(v)
            if j is not None:
                edges.append((i, j))
    return edges




IMAGENET = os.path.join(DATA_DIR_PREFIX, 'AWA2/materials', 'imagenet-split.json')
AWA = os.path.join(DATA_DIR_PREFIX, 'AWA2/materials', 'awa2-split.json')



# ImageNet
# js = json.load(open(IMAGENET, 'r'))
# imagenet_train = js['train']  # 1000

# awa
awa_js = json.load(open(AWA, 'r'))
awa_train = awa_js['train']  # 40
awa_test = awa_js['test']  # 10
awa_train_name = awa_js['train_names']
awa_test_name = awa_js['test_names']


# key_wnids = awa_test
#
# s = list(map(getnode, key_wnids))  # get wn_node
# print type(s[0])
# edges = getedges(s)

# wnids = list(map(getwnid, s))  # get s's wnids (graph nodes)


def getNeightborO(wnid):

    wnid_s = getnode(wnid)  # get wn synset
    print "wnid synset:", wnid_s

    for wn_per in wnid_s.hypernyms():
        print "hypernyms:", wn_per
        wnid = getwnid(wn_per)
        if wnid in awa_train:
            print wnid

    for wn_per in wnid_s.hyponyms():
        print "hyponyms:", wn_per
        wnid = getwnid(wn_per)
        if wnid in awa_train:
            print wnid

def getNeightbor(test_wn_name):
    wnid = awa_test[awa_test_name.index(test_wn_name)]
    print wnid
    wnid_s = getnode(wnid)  # get wn synset
    print "wnid synset:", wnid_s

    for wn_per in wnid_s.hypernyms():
        print "hypernyms:", wn_per
        wnid = getwnid(wn_per)
        if wnid in awa_train:
            print wnid

    for wn_per in wnid_s.hyponyms():
        print "hyponyms:", wn_per
        wnid = getwnid(wn_per)
        if wnid in awa_train:
            print wnid


def neighbor(sysnet):
    for wn_per in sysnet.hypernyms():
        wnid = getwnid(wn_per)
        if wnid in awa_train:
            print '- hypernyms + ', wnid, ' + ', wn_per

    for wn_per in sysnet.hyponyms():
        wnid = getwnid(wn_per)
        if wnid in awa_train:
            print '- hyponyms + ', wnid, ' + ', wn_per

def getNeightborN(test_wn_name):
    wnid = awa_test[awa_test_name.index(test_wn_name)]
    print wnid
    wnid_s = getnode(wnid)  # get wn synset
    print "wnid synset:", wnid_s

    for wn_per in wnid_s.hypernyms():
        print "hypernyms:", wn_per

        neighbor(wn_per)


    for wn_per in wnid_s.hyponyms():
        print "hyponyms:", wn_per

        neighbor(wn_per)


## get hypernyms
test_wn_name = 'sheep'
# getNeightbor(test_wn_name)
getNeightborN(test_wn_name)


##
# wnid = 'n02066245'
# getNeightborO(wnid)





