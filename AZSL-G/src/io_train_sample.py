
import argparse
import json
import numpy as np
import pickle as pkl
from scipy import sparse
import os

from tensorflow.python import pywrap_tensorflow

from io_graph import prepare_graph
'''
convert to gcn data: input, output, graph
'''


# save embedding vectors of all the vertices of the graph
def convert_input(wv_file, save_dir):
    with open(wv_file, 'rb') as fp:
        vertex_vectors = pkl.load(fp)
    vertex_vectors = vertex_vectors.tolist()
    sparse_vecs = sparse.csr_matrix(vertex_vectors)
    dense_vecs = np.array(vertex_vectors)

    sparse_file = os.path.join(save_dir, 'all_x.pkl')
    with open(sparse_file, 'wb') as fp:
        pkl.dump(sparse_vecs, fp)

    dense_file = os.path.join(save_dir, 'all_x_dense.pkl')  # embedding vectors of all vertices
    with open(dense_file, 'wb') as fp:
        pkl.dump(dense_vecs, fp)

    print('Save vectors of all vertices')

# label the graph nodes with seen classifiers

def convert_label(model_path, layer_name, save_dir):  # get output's label and mask
    ''' save visual classifier '''
    corresp_file = os.path.join(DATA_DIR, DATASET, 'corresp.json')
    with open(corresp_file) as fp:
        corresp_list = json.load(fp)

    def get_variables_in_checkpoint_file(file_name):
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map, reader

    # read feature weights for seen classes, ordered by class id
    var_keep_dic, reader = get_variables_in_checkpoint_file(model_path)
    for name in var_keep_dic:
        print(name, len(var_keep_dic[name]), var_keep_dic[name])
        if name == layer_name:
            print(name)
            print(reader.get_tensor(name).shape)
            fc = reader.get_tensor(name).squeeze()
            fc_dim = fc.shape[0]
            break
    print('fc:', fc.shape)  # (2048, 1000)
    # the position of a seen class has the vector of corresponding CNN feature weight
    # fc[:, class_id + offset] represents the feature weights of class_id
    fc_labels = np.zeros((len(corresp_list), fc_dim))
    print('fc dim ', fc_labels.shape)
    for i, corresp in enumerate(corresp_list):
        vertex_type = corresp[1]
        class_id = corresp[0]
        # seen class (vertex)
        if vertex_type == 0:
            fc_labels[i, :] = np.copy(fc[:, class_id])
            assert class_id < 398
    label_file = os.path.join(save_dir, 'train_y.pkl')
    with open(label_file, 'wb') as fp:
        pkl.dump(fc_labels, fp)

    # the position that is 1 means the vertex of that position is an unseen class
    test_index = []
    for corresp in corresp_list:
        if corresp[0] == -1:
            test_index.append(-1)
        else:
            test_index.append(corresp[1])  # corresp[1]: 0/1, the value 0 means seen class, 1 means unseen classes
    test_file = os.path.join(save_dir, 'test_index.pkl')
    with open(test_file, 'wb') as fp:
        pkl.dump(test_index, fp)


def convert_graph(save_dir):
    graph_file = os.path.join(DATA_DIR, DATASET, 'graph.pkl')
    if not os.path.exists(graph_file):
        prepare_graph()
    save_file = os.path.join(save_dir, 'graph.pkl')
    if os.path.exists(save_file):
        cmd = 'rm  %s' % save_file
        os.system(cmd)

    cmd = 'ln -s %s %s' % (graph_file, save_file)  # soft link
    os.system(cmd)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='../../data', help='root directory')
    parser.add_argument('--data_dir', type=str, default='AZSL-G', help='data directory')
    parser.add_argument('--dataset', type=str, default='ImNet_A', help='ImNet_A, AwA')

    args = parser.parse_args()

    DATA_DIR = os.path.join(args.data_root, args.data_dir)
    DATASET = args.dataset



    model_path = os.path.join(DATA_DIR, 'materials', 'resnet_v1_50.ckpt')
    layer_name = 'resnet_v1_50/logits/weights'

    wv_file = os.path.join(DATA_DIR, DATASET, 'glove_w2v.pkl')

    save_dir = os.path.join(DATA_DIR, DATASET, 'glove_res50')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    print('Converting input')
    convert_input(wv_file, save_dir)

    print('Converting graph')
    convert_graph(save_dir)

    print('Converting label')
    convert_label(model_path, layer_name, save_dir)  # label the graph nodes with seen classifiers
    print('Prepared data to %s' % save_dir)
