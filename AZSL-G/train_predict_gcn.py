from __future__ import division
from __future__ import print_function

import time
import os
import tensorflow as tf
import numpy as np
import pickle as pkl

import argparse


from model.utils import load_data_vis_multi
from model.utils import preprocess_features_dense2
from model.utils import preprocess_adj
from model.utils import create_config_proto
from model.utils import construct_feed_dict
from model.gcn import GCN_dense_mse

# from IMAGENET_Animal.Exp_Test.test_in_train import test_imagenet_zero

'''
train gcn
'''


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


parser = argparse.ArgumentParser()

parser.add_argument('--data_root', type=str, default='/home/gyx/X-ZSL/data', help='root directory')
parser.add_argument('--data_dir', type=str, default='AZSL-G', help='data directory')
parser.add_argument('--dataset', type=str, default='ImNet_A', help='ImNet_A, AwA')

parser.add_argument('--model', type=str, default='dense', help='Model type')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=600, help='Number of epochs to train.')
parser.add_argument('--save_epoch', type=int, default=300, help='epochs to save model.')
parser.add_argument('--hidden1', type=int, default=2048, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=2048, help='Number of units in hidden layer 2.')
parser.add_argument('--hidden3', type=int, default=1024, help='Number of units in hidden layer 3.')
parser.add_argument('--hidden4', type=int, default=1024, help='Number of units in hidden layer 4.')
parser.add_argument('--hidden5', type=int, default=512, help='Number of units in hidden layer 5.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--early_stopping', type=int, default=10, help='Tolerance for early stopping (# of epochs).')
parser.add_argument('--max_degree', type=int, default=3, help='Maximum Chebyshev polynomial degree.')
parser.add_argument('--gpu', type=str, default='1', help='gpu id.')
args = parser.parse_args()

DATA_DIR = os.path.join(args.data_root, args.data_dir)
DATASET = args.dataset
EXP_NAME = 'Exp1_GCN'

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
# adj, X, y_train, y_val, y_trainval, train_mask, val_mask, trainval_mask = \

# train_adj_mask, val_adj_mask is mainly for mask the attention weights matrices
adj, X, y_train, train_mask, train_adj_mask, val_mask, val_adj_mask, trainval_mask, trainval_adj_mask =\
        load_data_vis_multi(Input_File, use_trainval, X_dense_file, train_y_file,
                            graph_file, test_index_file)

# Some preprocessing
X, div_mat = preprocess_features_dense2(X)

if args.model == 'dense':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN_dense_mse
else:
    raise ValueError('Invalid argument for model: ' + str(args.model))


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
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'learning_rate': tf.placeholder(tf.float32, shape=())
}

# Create model
model = model_func(args, placeholders, input_dim=X.shape[1], logging=True)

sess = tf.Session(config=create_config_proto())

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# save_epochs = [300, 400, 500]

if not os.path.exists(out_path):
    os.makedirs(out_path)
    print('!!! Make directory %s' % out_path)
else:
    print('### save to: %s' % out_path)

# Train model
now_lr = args.learning_rate
for epoch in range(args.epochs):
    t = time.time()

    # Construct feed dictionary
    # train_mask: point out which vertices are used as seen classes
    # feed_dict = construct_feed_dict(X, support, y_train, train_mask, placeholders)
    # feed_dict.update({placeholders['learning_rate']: now_lr})
    feed_dict = construct_feed_dict(X, support, y_train, train_mask, train_adj_mask,
                                    val_mask, val_adj_mask, trainval_mask, trainval_adj_mask, placeholders)
    feed_dict.update({placeholders['learning_rate']: now_lr})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.optimizer._lr], feed_dict=feed_dict)

    if epoch % 20 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_loss_nol2=", "{:.5f}".format(outs[2]),
              "time=", "{:.5f}".format(time.time() - t),
              "lr=", "{:.5f}".format(float(outs[3])))
        # print(sess.run(model.outputs, feed_dict=feed_dict))

    # Predicting step
    # --save outputs
    if (epoch + 1) >= args.save_epoch and (epoch + 1) % 100 == 0:
        outs = sess.run(model.outputs, feed_dict=feed_dict)
        filename = os.path.join(out_path, ('feat_%d' % (epoch+1)))
        print(time.strftime('[%X %x %Z]\t') + 'save to: ' + filename)

        filehandler = open(filename, 'wb')
        pkl.dump(outs, filehandler)
        filehandler.close()

    

    # -- while training, while testing
    # if (epoch + 1) >= 300 and (epoch + 1) % 100 == 0:
    # # if (epoch + 1) % 100 == 0:
    #     outs = sess.run(model.outputs, feed_dict=feed_dict)
    #     # test
    #     result = test_imagenet_zero(weight_pred=outs)
    #
    #     output = ['{:.2f}'.format(i * 100) for i in result[0]]
    #     print('-----------Testing: Epoch = ', (epoch + 1), ' | accuracy = ',
    #           output)


print("Optimization Finished!")

sess.close()
