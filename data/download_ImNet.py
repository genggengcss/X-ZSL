import argparse
import os
import threading
import urllib
import glob

import cv2
import numpy as np


'''
download original images of ImageNet with class list 'ImNet_A_unseen.txt'
'''



def download(vid_file):
    with open(vid_file) as fp:
        vid_list = [line.strip() for line in fp]
    url_list = 'http://www.image-net.org/download/synset?wnid='
    url_key = '&username=%s&accesskey=%s&release=latest&src=stanford' % (args.user, args.key)

    testfile = urllib.URLopener()
    for i in range(len(vid_list)):
        wnid = vid_list[i]
        url_acc = url_list + wnid + url_key

        save_dir = os.path.join(SAVE_PATH, wnid)  # the path to save image
        lockname = save_dir + '.lock'
        if os.path.exists(save_dir):
            continue
        if os.path.exists(lockname):
            continue
        try:
            os.makedirs(lockname)
        except:
            continue
        tar_file = os.path.join(SAVE_PATH, wnid + '.tar')
        try:
            testfile.retrieve(url_acc, tar_file)
            print('Downloading %s' % wnid)
        except:
            print('!!! Error when downloading', wnid)
            continue

        if not os.path.exists(os.path.join(SAVE_PATH, wnid)):
            os.makedirs(os.path.join(SAVE_PATH, wnid))
        cmd = 'tar -xf ' + tar_file + ' --directory ' + save_dir
        os.system(cmd)
        cmd = 'rm ' + os.path.join(tar_file)
        os.system(cmd)
        cmd = 'rm -r %s' % lockname
        os.system(cmd)

        if i % 1 == 0:
            print('%d / %d' % (i, len(vid_list)))


def rm_empty(vid_file):
    with open(vid_file) as fp:
        vid_list = [line.strip() for line in fp]
    cnt = 0
    for i in range(len(vid_list)):
        save_dir = os.path.join(SAVE_PATH, vid_list[i])
        jpg_list = glob.glob(save_dir + '/*.JPEG')
        if len(jpg_list) < 10:
            print(vid_list[i])
            cmd = 'rm -r %s ' % save_dir
            os.system(cmd)
            cnt += 1
    print(cnt)

def down_sample(list_file, image_dir, size=256):
    with open(list_file) as fp:
        index_list = [line.split()[0] for line in fp]
    for i, index in enumerate(index_list):
        img_file = os.path.join(image_dir, index)
        if not os.path.exists(img_file):
            print('not exist:', img_file)
            continue
        img = downsample_image(img_file, size)
        if img is None:
            continue
        save_file = os.path.join(os.path.dirname(img_file), os.path.basename(img_file).split('.')[0] + 'copy') + '.JPEG'
        cv2.imwrite(save_file, img)
        cmd = 'mv %s %s' % (save_file, img_file)
        os.system(cmd)
        if i % 1000 == 0:
            print(i, len(index_list), index)

def downsample_image(img_file, target_size):
    img = cv2.imread(img_file)
    if img is None:
        return img
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    im_scale = min(1, im_scale)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    return img





if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('--data_root', type=str, default='', help='root directory')
    parser.add_argument('--dataset', type=str, default='ImNet_A', help='ImNet_A')
    parser.add_argument('--user', type=str, default='',
                        help='your username', required=False)
    parser.add_argument('--key', type=str, default='',
                        help='your access key', required=False)
    args = parser.parse_args()

    # the dir to save images
    # SAVE_PATH = os.path.join(args.data_root, 'images', args.dataset)
    SAVE_PATH = os.path.join('images', args.dataset)

    # class list to be downloaded
    # class_list_file = os.path.join(args.data_root, 'ImNet_A_unseen.txt')
    class_list_file = os.path.join('ImNet_A_unseen.txt')


    download(class_list_file)

