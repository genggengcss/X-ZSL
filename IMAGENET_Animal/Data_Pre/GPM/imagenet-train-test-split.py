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


# read txt file
def readTxtFile(file):
    lines = list()
    nodes = open(file, 'rU')
    try:
        for line in nodes:
            line = line[:-1]
            lines.append(line)
    finally:
        nodes.close()
    return lines





img_animal_split = os.path.join(DATA_DIR_PREFIX, 'IMAGENET_Animal/materials', 'imagenet-animal-split.json')
train_test_split = os.path.join(DATA_DIR_PREFIX, 'IMAGENET_Animal/materials', 'imagenet-train-test-split.json')
test_wnid_file = os.path.join(DATA_DIR_PREFIX, 'IMAGENET_Animal/materials', 'unseen_2012_2-hops_animal.txt')

if __name__ == '__main__':





    # imagenet split
    js = json.load(open(img_animal_split, 'r'))
    # print js
    train_wnids = js['train']  # 398
    test_wnids = js['test']  # 3395

    print len(js)
    print len(train_wnids)
    print len(test_wnids)
    test_wnid_2_hops = readTxtFile(test_wnid_file)

    test_wnid = list()
    for wnid in test_wnids:
        if wnid in test_wnid_2_hops:
            test_wnid.append(wnid)





    obj = {'train': train_wnids, 'test': test_wnid}
    json.dump(obj, open(train_test_split, 'w'))



