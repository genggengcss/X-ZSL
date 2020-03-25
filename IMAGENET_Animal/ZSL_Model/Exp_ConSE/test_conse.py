
# coding=gbk
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import cv2

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets import inception_v1
from tensorflow.contrib.slim.python.slim.nets.inception_v1 import inception_v1_base
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.python.ops import variable_scope
import time
import pickle as pkl
'''
extract feature of test images
with the help of 
'''



# DATA_DIR_PREFIX = '/Users/geng/Data/Human_X_ZSL_DATA/'
DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'
EXP_NAME = 'IMAGENET_Animal/Exp_ConSE'


def init(model_path, sess):
    def get_variables_in_checkpoint_file(file_name):
        reader = tf.pywrap_tensorflow.NewCheckpointReader(file_name)
        # reader.get_tensor()
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map, reader

    var_keep_dic, reader = get_variables_in_checkpoint_file(model_path)
    my_var_list = tf.global_variables()
    sess.run(tf.variables_initializer(my_var_list, name='init'))
    variables_to_restore = []
    my_dict = {}
    for v in my_var_list:
        name = v.name.split(':')[0]
        my_dict[name] = 0
        if name not in var_keep_dic:
            print('He does not have', name)
        else:
            if v.shape != var_keep_dic[name]:
                print('Does not match shape: ', v.shape, var_keep_dic[name])
                continue
            variables_to_restore.append(v)
    for name in var_keep_dic:
        if name not in my_dict:
            # print('I do not have ', name)
            continue
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, model_path)
    print('Initialized')

# 图片预处理
def preprocess_res50(image_name):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    image = cv2.imread(image_name)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 颜色空间转换函数：opencv中，图像不是用常规的RGB颜色通道来存储，而是用BGR顺序
    target_size = 256
    crop_size = 224
    im_size_min = np.min(image.shape[0:2])   # 取三个维度中的最小值，即通道数，这些图片的通道数为 3
    im_scale = float(target_size) / float(im_size_min)

    # cv2.resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None)
    # 参数输入：src原图片（宽*高*通道），dsize：输出图像尺寸；fx:沿水平轴的比例因子；fy:沿垂直轴的比例因子；interpolation：插值方法
    image = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    height = image.shape[0]  # 图像的垂直像素
    width = image.shape[1]   # 图像的 水平像素
    x = int((width - crop_size) / 2)
    y = int((height - crop_size) / 2)
    image = image[y: y + crop_size, x: x + crop_size]  # 剪裁图片

    image = image.astype(np.float32)
    image[:, :, 0] -= _R_MEAN
    image[:, :, 1] -= _G_MEAN
    image[:, :, 2] -= _B_MEAN
    image = image[np.newaxis, :, :, :]   # np.newaxis 插入新维度
    return image


def resnet_arg_scope(is_training=True,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }
    with arg_scope(
            [slim.conv2d],
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc




def run_pred(sess, pool5, image_holder, image):
    feat = sess.run(pool5, feed_dict={image_holder: image})
    # feat = np.squeeze(feat, axis=[1, 2])  # 使 feat变成一个 list
    # exit()
    return feat

def res50():
    image = tf.placeholder(tf.float32, [None, 224, 224, 3], 'image')
    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net_conv, end_point = resnet_v1.resnet_v1_50(image, num_classes=1000, global_pool=True, is_training=False)
    return net_conv, image




def extract_feature(pool5, image_holder, resnet_model_file, image_dir, test_image_file):
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    init(resnet_model_file, sess)
    print('Done Init! ')

    # get class's word embedding
    with open(class_word2vec_file, 'rb') as fp:  # glove
        word2vec_feat = pkl.load(fp)

    # read test image list
    image_list, label_list = [], []
    with open(test_image_file) as fp:
        for line in fp.readlines():
            index, label = line.split()
            image_list.append(index)  # list of images
            label_list.append(int(label))

    # extract testing class word embedding
    invalid_wv = 0
    test_class_matrix, test_class_index = [], []
    for k in range(1000, len(word2vec_feat)):
        t_wv = word2vec_feat[k]
        if np.linalg.norm(t_wv) == 0:  # 求范数
            invalid_wv = invalid_wv + 1
            continue
        test_class_matrix.append(t_wv)
        test_class_index.append(k)

    test_class_matrix = np.array(test_class_matrix)  # [497, 300]
    print('skip candidate class due to no word embedding: %d / %d:' % (invalid_wv, len(test_class_index) + invalid_wv))
    print('candidate class shape: ', test_class_matrix.shape)

    test_class_matrix = test_class_matrix.T  # [300, 497]
    test_class_index = np.array(test_class_index)  # [497]

    # remove invalid training class
    invalid_wv = 0
    seen_embed, seen_index = [], []
    for k in range(0, 1000):
        t_wv = word2vec_feat[k]
        if np.linalg.norm(t_wv) == 0:  # 求范数
            invalid_wv = invalid_wv + 1
            continue
        seen_embed.append(t_wv)
        seen_index.append(k)

    seen_embed = np.array(seen_embed)  # [1000, 300]
    seen_index = np.array(seen_index)
    print('skip candidate class due to no word embedding: %d / %d:' % (invalid_wv, len(seen_index) + invalid_wv))

    hitKs = [1]
    hit_retrv = [1, 2, 5, 10, 20]
    hit_count = np.zeros((len(hitKs), len(hit_retrv)))
    cnt_valid = 0  # count test images



    # prediction
    for i, index in enumerate(image_list):


        if label_list[i] not in test_class_index:  # remove invalid unseen class
            continue

        cnt_valid = cnt_valid + 1

        image_path = os.path.join(image_dir, index)
        image = preprocess_res50(image_path)  # 预处理图片  image shape:[None, 224, 224, 3]
        if image is None:
            print('no image')
            continue
        # predict image label: range(0-999)
        feat = run_pred(sess, pool5, image_holder, image)

        feat = tf.squeeze(feat, axis=[1, 2])

        prob = tf.nn.softmax(feat)

        pred = tf.cast(tf.argmax(feat, axis=1), dtype=tf.int32)
        # top_T = 10
        pred_K = tf.nn.top_k(feat, top_T).indices  # [[31 32 30]]
        prob_K = tf.nn.top_k(prob, top_T).values  # [[0.9794382  0.01893266 0.00162807]]

        prob_K = tf.nn.softmax(prob_K)

        pred_K_numpy = pred_K.eval(session=sess)
        prob_K_numpy = prob_K.eval(session=sess)


        # print(prob_K_numpy[0])  # [0.9794382  0.01893266 0.00162807]
        # print ("prediction prob:", pred_K_numpy[0])
        # print ("prediction prob:", prob_K_numpy[0])

        # embedding matrix
        embed_list = list()
        prob_list = list()
        for j in range(len(pred_K_numpy[0])):
            index = pred_K_numpy[0][j]

            if index in seen_index:
                embed = word2vec_feat[index]
                embed_list.append(embed)
                prob_list.append(prob_K_numpy[0][j])

        embed_array = np.array(embed_list)  # [10, 300]
        prob_array = np.array(prob_list) # [10]




        prob_m = prob_array[np.newaxis, :]  # [1, 10]

        conv_embed = np.dot(prob_m, embed_array)  # [1, 300]

        conv_embed = np.true_divide(np.squeeze(conv_embed), np.sum(np.squeeze(prob_m)))  # [300]


        # compute similarity
        conv_embed = conv_embed[np.newaxis, :]  # [1, 300]
        scores = np.dot(conv_embed, test_class_matrix).squeeze()  # [497]
        # print("similarity shape:", scores.shape)



        scores = scores - scores.max()
        scores = np.exp(scores)
        scores = scores / scores.sum()

        ids = np.argsort(-scores)

        for k in range(len(hitKs)):
            for m in range(len(hit_retrv)):
                current_len = hit_retrv[m]

                for sort_id in range(current_len):
                    label = test_class_index[ids[sort_id]]
                    if int(label) == label_list[i]:
                        hit_count[k][m] = hit_count[k][m] + 1
                        break

        if i % 100 == 0:

            print('processing %d / %d ' % (i, len(label_list)))

    hit_count = hit_count * 1.0 / cnt_valid
    # print(hit_count)
    print('total: %d', cnt_valid)

    output = ['{:.2f}'.format(i * 100) for i in hit_count[0]]

    print('----------------------')
    print('result: ', output)
    print('----------------------')






top_T = 10
class_word2vec_file = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'glove_word2vec.pkl')
resnet_model_file = os.path.join(DATA_DIR_PREFIX, 'materials', 'resnet_v1_50.ckpt')
image_dir = os.path.join(DATA_DIR_PREFIX, 'Test_DATA_feats/Test_DATA_feats/')
test_image_file = os.path.join(DATA_DIR_PREFIX, 'materials', 'test_img_list_2_hops_animal_less100.txt')

if __name__ == '__main__':



    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # resnet for predicting image to seen class (1000)
    pool5, image_holder = res50()


    extract_feature(pool5, image_holder, resnet_model_file, image_dir, test_image_file)
