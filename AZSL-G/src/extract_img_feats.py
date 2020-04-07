

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
'''
extract feature of test images using resnet pre-train model
'''


def extract_feature(image_list, pool5, image_holder, preprocess, model_path, image_dir, feat_dir):
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    init(model_path, sess)
    print('Done Init! ')
    net_time, cnt = 0, 0
    for i, index in enumerate(image_list):
        feat_name = os.path.join(feat_dir, index.split('.')[0] + '.npz')
        image_name = os.path.join(image_dir, index)
        lockname = feat_name + '.lock'
        if os.path.exists(feat_name):
            continue
        if os.path.exists(lockname):
            continue
        try:
            os.makedirs(lockname)
        except:
            continue
        t = time.time()
        cnt += 1

        image = preprocess(image_name)  #
        if image is None:
            print('no image')
            continue
        feat = run_feat(sess, pool5, image_holder, image)
        if not os.path.exists(os.path.dirname(feat_name)):
            try:
                os.makedirs(os.path.dirname(feat_name))
                print('## Make Directory: %s' % feat_name)
            except:
                pass
        np.savez_compressed(feat_name, feat=feat)
        net_time += time.time() - t
        if i % 1000 == 0:
            print('extracting feature [%d / %d] %s (%f sec)' % (i, len(image_list), feat_name, net_time / cnt * 1000),
                  feat.shape)
            net_time = 0
            cnt = 0
        cmd = 'rm -r %s' % lockname
        os.system(cmd)


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
            print('I do not have ', name)
    restorer = tf.train.Saver(variables_to_restore)
    restorer.restore(sess, model_path)
    print('Initialized')

# image preprocess
def preprocess_res50(image_name):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    image = cv2.imread(image_name)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target_size = 256
    crop_size = 224
    im_size_min = np.min(image.shape[0:2])   # number of image channels
    im_scale = float(target_size) / float(im_size_min)

    # cv2.resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None)
    # input: orignal image (w*h*c); dsize: output size; fx: ; fy: ; interpolation: ;
    image = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    height = image.shape[0]
    width = image.shape[1]
    x = int((width - crop_size) / 2)
    y = int((height - crop_size) / 2)
    image = image[y: y + crop_size, x: x + crop_size]  # crop image

    image = image.astype(np.float32)
    image[:, :, 0] -= _R_MEAN
    image[:, :, 1] -= _G_MEAN
    image[:, :, 2] -= _B_MEAN
    image = image[np.newaxis, :, :, :]
    return image



def run_feat(sess, pool5, image_holder, image):
    feat = sess.run(pool5, feed_dict={image_holder: image})
    feat = np.squeeze(feat)
    # exit()
    return feat


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




def res50():
    image = tf.placeholder(tf.float32, [None, 224, 224, 3], 'image')
    with slim.arg_scope(resnet_arg_scope(is_training=False)):
        net_conv, end_point = resnet_v1.resnet_v1_50(image, global_pool=True, is_training=False)
    return net_conv, image




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../../data', help='root directory')
    parser.add_argument('--data_dir', type=str, default='AZSL-G', help='data directory')
    parser.add_argument('--dataset', type=str, default='ImNet_A', help='ImNet_A, AwA')
    parser.add_argument('--gpu', type=str, default='1',
                        help='gpu device')
    args = parser.parse_args()


    DATA_DIR = os.path.join(args.data_root, args.data_dir)
    DATASET = args.dataset
    # pre-train model
    MODEL_PATH = os.path.join(DATA_DIR, 'materials', 'resnet_v1_50.ckpt')
    IMAGE_FILE = os.path.join(DATA_DIR, DATASET, 'test_img_list.txt')

    if DATASET == 'ImNet_A':
        Image_DIR = os.path.join(args.data_root, 'images', 'ImNet_A')
    if DATASET == 'AwA':
        Image_DIR = os.path.join(args.data_root, 'images', 'Animals_with_Attributes2/JPEGImages')

    SAVE_DIR = os.path.join(DATA_DIR, DATASET, 'Test_DATA_feats')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # load cnn model
    pool5, image_holder = res50()
    preprocess = preprocess_res50

    image_list, label_list = [], []
    with open(IMAGE_FILE) as fp:
        for line in fp.readlines():
            index, label = line.split()
            image_list.append(index)   # list of images
            label_list.append(int(label))

    extract_feature(image_list, pool5, image_holder, preprocess, MODEL_PATH, Image_DIR, SAVE_DIR)
