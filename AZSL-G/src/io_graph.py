import os
import json
import pickle as pkl
import argparse
import xml.etree.ElementTree as ET
'''
prepare_graph(): construct graph using wordnet tree and prepare node names
make_corresp_awa(): mark the class type and index in graph (seen, unseen and others)
'''



animal_wnid = 'n00015388'  # extract animal subset

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
    inv_wordn_file = os.path.join(DATA_DIR, DATASET, 'invdict_wordn.json')
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

    graph_file = os.path.join(DATA_DIR, DATASET, 'graph.pkl')
    with open(graph_file, 'wb') as fp:
        pkl.dump(graph, fp)
        print('Save ImageNet structure to: ', graph_file)

    # read words of each wnid
    wnid_word = dict()
    with open(os.path.join(DATA_DIR, 'materials', 'words.txt'), 'rb') as fp:
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
    inv_wordn_word_file = os.path.join(DATA_DIR, DATASET, 'invdict_wordntext.json')
    with open(inv_wordn_word_file, 'w') as fp:
        json.dump(words, fp)
        print('Save graph node in text to %s' % inv_wordn_word_file)


# get WordNet graph
# save vertices, words of each vertex, adjacency matrix of the graph
def prepare_graph():
    structure_file = os.path.join(DATA_DIR, 'materials', 'wordnet_tree_structure.xml')
    tree = ET.parse(structure_file)
    root = tree.getroot()
    # move to the animal nodes start
    for sy in root.findall('synset'):  # find synset tag
        for ssy in sy.findall('synset'):  # deeper layer
            print("wnid:", ssy.get('wnid'))
            if ssy.get('wnid') == animal_wnid:  # get tag's attribute value(wnid), 'n00015388' represents 'animal'
                vertex_list, edge_list = add_edge_dfs(ssy)  # animal node -> the root node
            else:
                continue
    # move to the animal nodes end

    vertex_list = list(set(vertex_list))  # remove the repeat node
    print('Unique Vertex #: %d, Edge #: %d', (len(vertex_list), len(edge_list)))
    convert_to_graph(vertex_list, edge_list)


# Label each seen and unseen class with an order number e.g., 0-397, 398-407
# if a graph vertex is a seen or unseen class
#   set the vertex with the class id and label of seen (value: 0) or unseen (1)
def make_corresp():
    seen_file = os.path.join(DATA_DIR, DATASET, 'seen.txt')  # nun:398
    unseen_file = os.path.join(DATA_DIR, DATASET, 'unseen.txt')  # num: 497
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

    inv_wordn_file = os.path.join(DATA_DIR, DATASET, 'invdict_wordn.json')
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
            assert corresp[0] >= 398
        elif corresp[1] == 0:
            check_train += 1
            assert 0 <= corresp[0] < 398
        else:
            assert corresp[0] == -1
    print('unseen classes #: %d, seen classes #: %d' % (check_test, check_train))

    save_file = os.path.join(DATA_DIR, DATASET, 'corresp.json')
    with open(save_file, 'w') as fp:
        json.dump(corresp_list, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='/home/gyx/X-ZSL/data', help='root directory')
    parser.add_argument('--data_dir', type=str, default='AZSL-G', help='data directory')
    parser.add_argument('--dataset', type=str, default='ImNet_A', help='ImNet_A, AwA')

    args = parser.parse_args()

    DATA_DIR = os.path.join(args.data_root, args.data_dir)
    DATASET = args.dataset


    prepare_graph()
    make_corresp()

