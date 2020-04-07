import argparse
import glob
import os




# create the list file of testing images and labels
def make_image_list(offset):
    with open(class_list_file) as fp:
        wnid_list = [line.strip() for line in fp]
    print(len(wnid_list))


    wr_fp = open(save_file, 'w')
    for i, wnid in enumerate(wnid_list):
        if DATASET == 'ImNet_A':
            img_list = glob.glob(os.path.join(Image_Path, wnid, '*.JPEG'))
        if DATASET == 'AwA':
            img_list = glob.glob(os.path.join(Image_Path, wnid, '*.jpg'))

        for path in img_list:
            index = os.path.join(wnid, os.path.basename(path))
            l = i + offset
            wr_fp.write('%s %d\n' % (index, l))
        if len(img_list) == 0:
            print('Warning: does not have class %s. Do you forgot to download the picture??' % wnid)
    wr_fp.close()




if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../../data', help='root directory')
    parser.add_argument('--data_dir', type=str, default='AZSL-G', help='data directory')
    parser.add_argument('--dataset', type=str, default='ImNet_A', help='ImNet_A, AwA')
    # parser.add_argument('--image_path', type=str, default='', help='the path to store images')
    parser.add_argument('--offset', type=int, default=398, help='the bais of label index')

    args = parser.parse_args()

    DATA_DIR = os.path.join(args.data_root, args.data_dir)
    DATASET = args.dataset

    if DATASET == 'ImNet_A':
        Image_Path = os.path.join(args.data_root, 'images', 'ImNet_A')
    if DATASET == 'AwA':
        Image_Path = os.path.join(args.data_root, 'images', 'Animals_with_Attributes2/JPEGImages')


    class_list_file = os.path.join(DATA_DIR, DATASET, 'unseen.txt')
    save_file = os.path.join(DATA_DIR, DATASET, 'test_img_list.txt')

    make_image_list(args.offset)

