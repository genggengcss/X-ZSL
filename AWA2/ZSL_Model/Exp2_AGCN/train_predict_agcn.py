# coding=gbk
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import time
import os
import tensorflow as tf
import numpy as np
import pickle as pkl



import sys
sys.path.append('../../../')

from ZSL.agcn.utils import load_data_vis_multi
from ZSL.agcn.utils import preprocess_features_dense2
from ZSL.agcn.utils import preprocess_adj
from ZSL.agcn.utils import adj_to_bias
from ZSL.agcn.utils import create_config_proto
from ZSL.agcn.utils import construct_feed_dict
from ZSL.agcn.models import GCN_dense_mse

from test_in_train import test_awa_zero
'''
    agcn model
'''


DATA_DIR_PREFIX = 'home/gyx/Data/KG_SS_DATA/ZSL/'
# DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'
EXP_DATA = 'AWA2/Exp1_GCN'
EXP_NAME = 'AWA2/Exp2_AGCN'

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', os.path.join(DATA_DIR_PREFIX, EXP_DATA, 'glove_res50'), 'Dataet string.')
flags.DEFINE_string('model', 'dense', 'Model string.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_string('out_path', os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'glove_res50/output_atten'), 'save dir')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 2048, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 2048, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden3', 1024, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden4', 1024, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden5', 512, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-3, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('gpu', '0', 'gpu id')
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

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
        load_data_vis_multi(FLAGS.dataset, use_trainval, X_dense_file, train_y_file,
                            graph_file, test_index_file)

# Some preprocessing
X, div_mat = preprocess_features_dense2(X)

if FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN_dense_mse

    adj = adj.todense()  # Return a dense matrix representation of this matrix
    biases_mask = adj_to_bias(adj, nhood=1)
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

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
model = model_func(placeholders, input_dim=X.shape[1], node_num=X.shape[0], logging=True)

sess = tf.Session(config=create_config_proto())

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

save_epochs = [1000, 1200, 1500, 1800, 2000]

if not os.path.exists(FLAGS.out_path):
    os.makedirs(FLAGS.out_path)
    print('!!! Make directory %s' % FLAGS.out_path)
else:
    print('### save to: %s' % FLAGS.out_path)

# Train model
now_lr = FLAGS.learning_rate
for epoch in range(FLAGS.epochs):
    t = time.time()

    # Construct feed dictionary
    # train_mask: point out which vertices are used as seen classes
    feed_dict = construct_feed_dict(X, support, y_train, train_mask, train_adj_mask,
                                    val_mask, val_adj_mask, trainval_mask, trainval_adj_mask, biases_mask, placeholders)
    feed_dict.update({placeholders['learning_rate']: now_lr})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.optimizer._lr], feed_dict=feed_dict)
    # outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    # trainval_adj_mask_f = tf.cast(trainval_adj_mask, dtype=tf.float32)
    # print(sess.run(trainval_adj_mask_f))
    if epoch % 20 == 0:
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_loss_nol2=", "{:.5f}".format(outs[2]),
              "time=", "{:.5f}".format(time.time() - t),
              "lr=", "{:.5f}".format(float(outs[3])))
        # print(sess.run(model.outputs, feed_dict=feed_dict))
        # print("trainval_adj_mask:", sess.run(tf.cast(trainval_adj_mask, dtype=tf.float32)))
        # print("train type:", type(tf.cast(trainval_adj_mask, dtype=tf.float32)))
        # print("biase type:", type(tf.cast(biases_mask, dtype=tf.float32)))
        # print("biase_mask:", sess.run(tf.cast(biases_mask, dtype=tf.float32)))


    # Predicting step
    if (epoch+1) in save_epochs:
        ###  save outputs
        outs = sess.run(model.outputs_atten, feed_dict=feed_dict)


        filename = os.path.join(FLAGS.out_path, ('feat_%d' % (epoch+1)))
        print(time.strftime('[%X %x %Z]\t') + 'save to: ' + filename)

        filehandler = open(filename, 'wb')
        pkl.dump(outs, filehandler)
        filehandler.close()

        ### save attention weights
        attens = sess.run(model.coefs, feed_dict=feed_dict)
        filename_atten = os.path.join(FLAGS.out_path, ('coef_%d' % (epoch+1)))
        filehandler = open(filename_atten, 'wb')
        pkl.dump(attens, filehandler)
        filehandler.close()

    # -- while training, while testing
    if epoch == 0:
        outs = sess.run(model.outputs_atten, feed_dict=feed_dict)
        # test
        result = test_awa_zero(weight_pred=outs)

        output = ['{:.2f}'.format(i * 100) for i in result[0]]
        print('-----------Testing: Epoch = ', (epoch + 1), ' | accuracy = ',
              output)

    if (epoch + 1) >= 800 and (epoch + 1) % 100 == 0:
        outs = sess.run(model.outputs_atten, feed_dict=feed_dict)
        # test
        result = test_awa_zero(weight_pred=outs)

        output = ['{:.2f}'.format(i * 100) for i in result[0]]
        print('-----------Testing: Epoch = ', (epoch + 1), ' | accuracy = ',
              output)

print("Optimization Finished!")

sess.close()
