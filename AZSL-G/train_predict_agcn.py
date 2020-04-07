from __future__ import division
from __future__ import print_function

import time
import os
import tensorflow as tf
import numpy as np
import pickle as pkl



import sys

import argparse


'''
    agcn model
'''



from model.utils import load_data_vis_multi
from model.utils import preprocess_features_dense2
from model.utils import preprocess_adj
from model.utils import create_config_proto
from model.utils import construct_feed_dict_agcn
from model.agcn import GCN_dense_mse
from model.utils import adj_to_bias
import val_agcn



# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


parser = argparse.ArgumentParser()

parser.add_argument('--data_root', type=str, default='../data', help='root directory')
parser.add_argument('--data_dir', type=str, default='AZSL-G', help='data directory')
parser.add_argument('--dataset', type=str, default='ImNet_A', help='ImNet_A, AwA')

parser.add_argument('--model', type=str, default='dense', help='Model type')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--save_epoch', type=int, default=1000, help='epochs to save model.')
parser.add_argument('--hidden1', type=int, default=2048, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=2048, help='Number of units in hidden layer 2.')
parser.add_argument('--hidden3', type=int, default=1024, help='Number of units in hidden layer 3.')
parser.add_argument('--hidden4', type=int, default=1024, help='Number of units in hidden layer 4.')
parser.add_argument('--hidden5', type=int, default=512, help='Number of units in hidden layer 5.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--early_stopping', type=int, default=10, help='Tolerance for early stopping (# of epochs).')
parser.add_argument('--max_degree', type=int, default=3, help='Maximum Chebyshev polynomial degree.')
parser.add_argument('--trainval', action='store_true', default=False, help='validation set')
parser.add_argument('--gpu', type=str, default='1', help='gpu id.')
args = parser.parse_args()

DATA_DIR = os.path.join(args.data_root, args.data_dir)
DATASET = args.dataset
EXP_NAME = 'Exp2_AGCN'

# the file to input GCN
Input_File = os.path.join(DATA_DIR, DATASET, 'glove_res50')
out_path = os.path.join(DATA_DIR, DATASET, EXP_NAME)  # save dir


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

use_trainval = True
X_dense_file = 'all_x_dense.pkl'
train_y_file = 'train_y.pkl'
graph_file = 'graph.pkl'  # not soft link
test_index_file = 'test_index.pkl'

# Load data
# '_train' represents vertices of seen classes (training)
# '_val' represents vertices of unseen classes (testing)
# '_trainval' = '_train' + '_val'
# '_mask' represents index, e.g., train_mask represents index of vertices of seen classes

# train_adj_mask, val_adj_mask is mainly for mask the attention weights matrices
adj, X, y_train, train_mask, train_adj_mask, val_mask, val_adj_mask, trainval_mask, trainval_adj_mask = \
        load_data_vis_multi(Input_File, use_trainval, X_dense_file, train_y_file,
                            graph_file, test_index_file)

# Some preprocessing
X, div_mat = preprocess_features_dense2(X)

if args.model == 'dense':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN_dense_mse

    adj = adj.todense()  # Return a dense matrix representation of this matrix
    biases_mask = adj_to_bias(adj, nhood=1)
else:
    raise ValueError('Invalid argument for model: ' + str(args.model))

print(X.shape)



# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],   # adj
    'features': tf.placeholder(tf.float32, shape=(X.shape[0], X.shape[1])),  # sparse_placeholder
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'labels_adj_mask': tf.placeholder(tf.int32),
    'val_mask': tf.placeholder(tf.int32),
    'val_adj_mask': tf.placeholder(tf.int32),
    'trainval_mask': tf.placeholder(tf.int32),
    'trainval_adj_mask': tf.placeholder(tf.int32),
    'biase_mask': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'learning_rate': tf.placeholder(tf.float32, shape=())
}

# Create model
model = model_func(args, placeholders, input_dim=X.shape[1], node_num=X.shape[0], logging=True)

sess = tf.Session(config=create_config_proto())

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []


if not os.path.exists(out_path):
    os.makedirs(out_path)
    print('!!! Make directory %s' % out_path)
else:
    print('### save to: %s' % out_path)

# Train model
now_lr = args.learning_rate
for epoch in range(1, args.epochs+1):
    t = time.time()

    # Construct feed dictionary
    # train_mask: point out which vertices are used as seen classes
    feed_dict = construct_feed_dict_agcn(X, support, y_train, train_mask, train_adj_mask,
                                    val_mask, val_adj_mask, trainval_mask, trainval_adj_mask, biases_mask, placeholders)
    feed_dict.update({placeholders['learning_rate']: now_lr})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.optimizer._lr], feed_dict=feed_dict)
    # outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    # trainval_adj_mask_f = tf.cast(trainval_adj_mask, dtype=tf.float32)
    # print(sess.run(trainval_adj_mask_f))
    if epoch % 20 == 0:
        print("Epoch:", '%04d' % epoch, "train_loss=", "{:.5f}".format(outs[1]),
              "train_loss_nol2=", "{:.5f}".format(outs[2]),
              "time=", "{:.5f}".format(time.time() - t),
              "lr=", "{:.5f}".format(float(outs[3])))


    # # Predicting step
    if epoch >= args.save_epoch and epoch % 100 == 0:
        ###  save outputs
        outs = sess.run(model.outputs_atten, feed_dict=feed_dict)


        filename = os.path.join(out_path, ('feat_%d' % epoch))
        print(time.strftime('[%X %x %Z]\t') + 'save to: ' + filename)
        filehandler = open(filename, 'wb')
        pkl.dump(outs, filehandler)
        filehandler.close()

        ### save attention weights
        attens = sess.run(model.coefs, feed_dict=feed_dict)
        filename_atten = os.path.join(out_path, ('coef_%d' % epoch))
        filehandler = open(filename_atten, 'wb')
        pkl.dump(attens, filehandler)
        filehandler.close()

    # validation
    if args.trainval:
        if epoch >= args.save_epoch and epoch % 50 == 0:
            outs = sess.run(model.outputs_atten, feed_dict=feed_dict)
            # test
            result = val_agcn.val(weight_pred=outs, dir=DATA_DIR, dataset=DATASET)

            output = ['{:.2f}'.format(i * 100) for i in result[0]]
            print('----------- Val: Epoch = ', epoch, ' | accuracy = ',
                  output)

print("Optimization Finished!")

sess.close()
