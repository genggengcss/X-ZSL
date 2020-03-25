import os
import json
import pickle as pkl

import xml.etree.ElementTree as ET
'''
prepare_list.py: prepare_graph(), make_corresp_awa()
'''


DATA_DIR_PREFIX = '/home/gyx/Data/KG_SS_DATA/ZSL/'
# DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'
EXP_NAME = 'IMAGENET/Exp1_GCN'


# add the edges between the input node and its children
# with deep first search
def add_edge_dfs(node):
    edges = []
    vertices = [node.attrib['wnid']]
    if len(node) == 0:
        return vertices, edges
    for child in node:
        if child.tag != 'synset':
            print(child.tag)

        edges.append((node.attrib['wnid'], child.attrib['wnid']))
        child_ver, child_edge = add_edge_dfs(child)
        edges.extend(child_edge)
        vertices.extend(child_ver)
    return vertices, edges


def convert_to_graph(vertices, edges):

    # save vertices
    inv_wordn_file = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'invdict_wordn.json')
    with open(inv_wordn_file, 'w') as fp:
        json.dump(vertices, fp)
        print('Save graph node in wnid to %s' % inv_wordn_file)

    # save the graph as adjacency matrix
    ver_dict = {}
    graph = {}
    for i, vertex in enumerate(vertices):
        ver_dict[vertex] = i
        graph[i] = []

    for edge in edges:
        id1 = ver_dict[edge[0]]
        id2 = ver_dict[edge[1]]
        graph[id1].append(id2)
        graph[id2].append(id1)

    graph_file = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'imagenet_graph.pkl')
    with open(graph_file, 'wb') as fp:
        pkl.dump(graph, fp)
        print('Save ImageNet structure to: ', graph_file)

    # read words of each wnid
    wnid_word = dict()
    with open(os.path.join(DATA_DIR_PREFIX, 'materials', 'words.txt'), 'rb') as fp:
        for line in fp.readlines():
            wn, name = line.split('\t')
            wnid_word[wn] = name.strip()

    # get and save words of each vertex
    words = []
    for wnid in vertices:
        if wnid in wnid_word:
            words.append(wnid_word[wnid])
        else:
            words.append('--%s--' % wnid)
            print('%s has no words' % wnid)
    inv_wordn_word_file = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'invdict_wordntext.json')
    with open(inv_wordn_word_file, 'w') as fp:
        json.dump(words, fp)
        print('Save graph node in text to %s' % inv_wordn_word_file)


# get WordNet graph
# save vertices, words of each vertex, adjacency matrix of the graph
def prepare_graph():
    structure_file = os.path.join(DATA_DIR_PREFIX, 'materials', 'wordnet_tree_structure.xml')
    tree = ET.parse(structure_file)
    root = tree.getroot()
    vertex_list, edge_list = add_edge_dfs(root[1])
    vertex_list = list(set(vertex_list))  # remove the repeat node
    print('Unique Vertex #: %d, Edge #: %d', (len(vertex_list), len(edge_list)))
    convert_to_graph(vertex_list, edge_list)


# Label each seen and unseen class with an order number e.g., 0-999, 1000-1009
# if a graph vertex is a seen or unseen class
#   set the vertex with the class id and label of seen (value: 0) or unseen (1)
def make_corresp_awa():
    seen_file = os.path.join(DATA_DIR_PREFIX, 'IMAGENET/materials', 'seen_2012_1k.txt')  # num: 1000
    unseen_file = os.path.join(DATA_DIR_PREFIX, 'IMAGENET/materials', 'unseen_2012_2-hops.txt')  # num:1549
    seen_dict = {}
    unseen_dict = {}
    cnt = 0
    with open(seen_file) as fp:
        for line in fp.readlines():
            seen_dict[line.strip()] = cnt
            cnt += 1

    with open(unseen_file) as fp:
        for line in fp.readlines():
            unseen_dict[line.strip()] = cnt
            cnt += 1

    inv_wordn_file = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'invdict_wordn.json')
    with open(inv_wordn_file) as fp:
        wnids = json.load(fp)

    corresp_list = []
    for wnid in wnids:
        # this is a seen class, label: 0
        if wnid in seen_dict:
            corresp_id = seen_dict[wnid]
            corresp_list.append([corresp_id, 0])
        # this is an unseen class, label: 1
        elif wnid in unseen_dict:
            corresp_id = unseen_dict[wnid]
            corresp_list.append([corresp_id, 1])
        else:
            corresp_list.append([-1, -1])

    check_train, check_test = 0, 0
    for corresp in corresp_list:
        if corresp[1] == 1:
            check_test += 1
            assert corresp[0] >= 1000
        elif corresp[1] == 0:
            check_train += 1
            assert 0 <= corresp[0] < 1000
        else:
            assert corresp[0] == -1
    print('unseen classes #: %d, seen classes #: %d' % (check_test, check_train))

    save_file = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'corresp-train-test.json')
    with open(save_file, 'w') as fp:
        json.dump(corresp_list, fp)


if __name__ == '__main__':
    prepare_graph()
    make_corresp_awa()

