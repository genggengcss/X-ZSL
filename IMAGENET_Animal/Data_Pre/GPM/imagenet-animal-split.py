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

def extractSub(wnids, animal):
    wnids_sub = list()
    for wnid in wnids:
        if wnid in animal:
            wnids_sub.append(wnid)
    print len(wnids_sub)

    return wnids_sub



animal_wnids_file = os.path.join(DATA_DIR_PREFIX, 'IMAGENET_Animal/materials', 'animal_nodes.txt')

imagenet_xml_file = os.path.join(DATA_DIR_PREFIX, 'materials', 'imagenet-xml-wnids.json')
imagenet_xml_animal_file = os.path.join(DATA_DIR_PREFIX, 'IMAGENET_Animal/materials', 'imagenet-xml-animal-wnids.json')


imagenet_split = os.path.join(DATA_DIR_PREFIX, 'materials', 'imagenet-split.json')
img_animal_split = os.path.join(DATA_DIR_PREFIX, 'IMAGENET_Animal/materials', 'imagenet-animal-split.json')

if __name__ == '__main__':



    # imagenet wnids, length: 32295
    xml_wnids = json.load(open(imagenet_xml_file, 'r'))
    print len(xml_wnids)

    # animal wnids
    animal_wnids = readTxtFile(animal_wnids_file)


    # imagenet animal subset
    xml_sub = extractSub(xml_wnids, animal_wnids)
    json.dump(xml_sub, open(imagenet_xml_animal_file, 'w'))



    # imagenet split
    js = json.load(open(imagenet_split, 'r'))
    # print js
    train_wnids = js['train']  # 1000
    test_wnids = js['test']  # 20842

    train_wnids_sub = extractSub(train_wnids, animal_wnids)
    test_wnids_sub = extractSub(test_wnids, animal_wnids)


    obj = {'train': train_wnids_sub, 'test': test_wnids_sub}
    json.dump(obj, open(img_animal_split, 'w'))





