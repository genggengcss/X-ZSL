# coding=gbk
# -*- coding: utf-8 -*-

import argparse
import re
import os
import json
import numpy as np
import pickle as pkl


"""
for extracting word embedding yourself, please download pretrained model from one of the following links.
"""
'''
original: obtain_word_embedding.py, for get embedding vector of vertives 
'''
url = {'glove': 'http://nlp.stanford.edu/data/glove.6B.zip',
       'google': 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing',
       'fasttext': 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip'}

WORD_VEC_LEN = 300

# # DATA_DIR_PREFIX = '/Users/geng/Data/Human_X_ZSL_DATA/'
# # DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'
# # EXP_NAME = 'IMAGENET_Animal/Exp1_GCN'
# Mat_DATA_DIR = '/home/gyx/X_ZSL/data/materials'
# DATA_DIR = '/home/gyx/X_ZSL/data/GCNZ'
# # DATASET = 'ImNet_A'
# DATASET = 'AwA'
# # EXP_NAME = 'Exp1_GCN'

def embed_text_file(vertices_f, word_vectors, get_vector, save_file):
    with open(vertices_f) as fp:
        vertices_list = json.load(fp)

    all_feats = []

    has = 0
    cnt_missed = 0
    missed_list = []
    for i, vertex in enumerate(vertices_list):
        class_name = vertex.lower()
        if i % 500 == 0:
            print('%d / %d : %s' % (i, len(vertices_list), class_name))
        feat = np.zeros(WORD_VEC_LEN)

        options = class_name.split(',')
        cnt_word = 0
        for option in options:
            now_feat = get_embedding(option.strip(), word_vectors, get_vector)
            if np.abs(now_feat.sum()) > 0:
                cnt_word += 1
                feat += now_feat
        if cnt_word > 0:
            feat = feat / cnt_word

        if np.abs(feat.sum()) == 0:
            print('cannot find word ' + class_name)
            cnt_missed = cnt_missed + 1
            missed_list.append(class_name)
        else:
            has += 1
            feat = feat / (np.linalg.norm(feat) + 1e-6)

        all_feats.append(feat)

    all_feats = np.array(all_feats)

    for each in missed_list:
        print(each)
    print('does not have semantic embedding: ', cnt_missed, 'has: ', has)

    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
        print('## Make Directory: %s' % save_file)
    with open(save_file, 'wb') as fp:
        pkl.dump(all_feats, fp)
    print('save to : %s' % save_file)


def get_embedding(entity_str, word_vectors, get_vector):
    try:
        feat = get_vector(word_vectors, entity_str)
        return feat
    except:
        feat = np.zeros(WORD_VEC_LEN)

    str_set = filter(None, re.split("[ \-_]+", entity_str))
    str_set = list(str_set)
    cnt_word = 0
    for i in range(len(str_set)):
        temp_str = str_set[i]
        try:
            now_feat = get_vector(word_vectors, temp_str)
            feat = feat + now_feat
            cnt_word = cnt_word + 1
        except:
            continue

    if cnt_word > 0:
        feat = feat / cnt_word
    return feat


def get_glove_dict(txt_dir):
    print('load glove word embedding')
    txt_file = os.path.join(txt_dir, 'glove.6B.300d.txt')
    word_dict = {}
    feat = np.zeros(WORD_VEC_LEN)
    with open(txt_file) as fp:
        for line in fp:
            words = line.split()
            assert len(words) - 1 == WORD_VEC_LEN
            for i in range(WORD_VEC_LEN):
                feat[i] = float(words[i+1])
            feat = np.array(feat)
            word_dict[words[0]] = feat
    print('loaded to dict!')
    return word_dict


def glove_google(word_vectors, word):
    return word_vectors[word]


def fasttext(word_vectors, word):
    return word_vectors.get_word_vector(word)


# transform the vectors of the graph (invdict_wordntext.json) to vectors
# save the vectors in e.g., glove_word2vec_wordnet.pkl
# the order of the vectors are consistent with the wordntext
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='/home/gyx/X_ZSL/data', help='root directory')
    parser.add_argument('--data_dir', type=str, default='GCNZ', help='data directory')
    parser.add_argument('--dataset', type=str, default='ImNet_A', help='ImNet_A, AwA')

    args = parser.parse_args()

    DATA_DIR = os.path.join(args.data_root, args.data_dir)
    DATASET = args.dataset


    vertices_file = os.path.join(DATA_DIR, DATASET, 'invdict_wordntext.json')

    save_file = os.path.join(DATA_DIR, DATASET, 'glove_w2v.pkl')
    if not os.path.exists(save_file):
        word_vectors = get_glove_dict(os.path.join(args.data_root))
        get_vector = glove_google

    print('obtain semantic word embedding', save_file)
    embed_text_file(vertices_file, word_vectors, get_vector, save_file)