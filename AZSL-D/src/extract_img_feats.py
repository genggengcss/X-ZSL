import argparse
import json
import os
import sys
import time
import torch
from torch.utils.data import DataLoader
import os.path as osp
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


sys.path.append('../')
from model import resnet
from model import utils
from model import image_folder






def extract_feat(path, class_name, cnn, feat_dir, stage='train'):

    # pre_process begin
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms_image = None
    if stage == 'train':
        transforms_image = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])
    if stage == 'test':
        transforms_image = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize])
    # pre_process end

    cls_path = osp.join(path, class_name)
    images = os.listdir(cls_path)
    for i, image in enumerate(images):

        image_path = osp.join(cls_path, image)
        # print image_path

        image = Image.open(image_path).convert('RGB')
        image = transforms_image(image)
        if image.shape[0] != 3 or image.shape[1] != 224 or image.shape[2] != 224:
            print('you should delete this guy:', image_path)



        path = image_path.replace('.JPEG', '.pth')

        feat_path = os.path.join(feat_dir, class_name, path.split('/')[-1])

        lock_path = feat_path + '.lock'
        if os.path.exists(feat_path):
            continue
        if os.path.exists(lock_path):
            continue
        try:
            os.makedirs(lock_path)
        except:
            continue

        # get features
        image = torch.unsqueeze(image, 0)
        # print '\n', image.shape
        feat = cnn(image)  # [1, 3, 224, 224]

        if not os.path.exists(os.path.dirname(feat_path)):
            try:
                os.makedirs(os.path.dirname(feat_path))
                print('## Make Directory: %s' % feat_path)
            except:
                pass



        torch.save(feat, feat_path)


        if i % 100 == 0:
            print('extracting feature [%d / %d]' % (i, len(images)))

        cmd = 'rm -r %s' % lock_path
        os.system(cmd)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='../../data', help='root directory')
    parser.add_argument('--data_dir', type=str, default='AZSL-D', help='data directory')
    parser.add_argument('--dataset', type=str, default='ImNet_A', help='ImNet_A, AwA')

    parser.add_argument('--stage', default='test')

    parser.add_argument('--gpu', type=str, default='1', help='gpu device')
    args = parser.parse_args()

    DATA_DIR = os.path.join(args.data_root, args.data_dir)
    DATASET = args.dataset

    CNN_Model = os.path.join(DATA_DIR, 'materials', 'resnet50-base.pth')

    SAVR_DIR = os.path.join(DATA_DIR, DATASET, 'Test_DATA_feats')

    if DATASET == 'ImNet_A':
        DATA_Split = os.path.join(DATA_DIR, DATASET, 'seen-unseen-split.json')
    if DATASET == 'AwA':
        DATA_Split = os.path.join(DATA_DIR, DATASET, 'awa2-split.json')





    # set_gpu(args.gpu)
    '''
    seen-unseen-split.json:
    [train], [test]
    '''

    split = json.load(open(DATA_Split, 'r'))
    train_wnids = split['train']
    print len(train_wnids)
    tests_wnids = split['test']
    print len(tests_wnids)

    ## load resnet model
    cnn = resnet.make_resnet50_base()
    cnn.load_state_dict(torch.load(CNN_Model))
    cnn.eval()


    # the directory of raw imagenet data, with images
    if DATASET == 'ImNet_A':
        images_path = os.path.join(args.data_root, 'images', 'ImNet_A')
    if DATASET == 'AwA':
        images_path = os.path.join(args.data_root, 'images', 'Animals_with_Attributes2/JPEGImages')



    for i, name in enumerate(tests_wnids, 1):

        print i, ' ', name, ' extracting features....',

        extract_feat(images_path, name, cnn, SAVR_DIR, stage=args.stage)





