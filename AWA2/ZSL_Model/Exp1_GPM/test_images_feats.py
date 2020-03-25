import argparse
import json
import os
import sys
import time
import torch
from torch.utils.data import DataLoader
import os.path as osp
# from PIL import Image
# import torchvision.transforms as transforms
from torch.utils.data import Dataset


sys.path.append('../../../../')
from ZSL.gpm import resnet
# from ZSL.gpm import utils
# from ZSL.gpm import image_folder




# DATA_DIR_PREFIX = '/home/gyx/Data/KG_SS_DATA/ZSL/'
DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'
EXP_NAME = 'AWA2/Exp1_GPM'





# def extract_feat(path, class_name, cnn, feat_dir, stage='train'):
#
#     # pre_process begin
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     transforms_image = None
#     if stage == 'train':
#         transforms_image = transforms.Compose([transforms.RandomResizedCrop(224),
#                                               transforms.RandomHorizontalFlip(),
#                                               transforms.ToTensor(),
#                                               normalize])
#     if stage == 'test':
#         transforms_image = transforms.Compose([transforms.Resize(256),
#                                               transforms.CenterCrop(224),
#                                               transforms.ToTensor(),
#                                               normalize])
#     # pre_process end
#
#     cls_path = osp.join(path, class_name)
#     images = os.listdir(cls_path)
#     for i, image in enumerate(images):
#
#         image_path = osp.join(cls_path, image)
#         # print image_path
#
#         image = Image.open(image_path).convert('RGB')
#         image = transforms_image(image)
#         if image.shape[0] != 3 or image.shape[1] != 224 or image.shape[2] != 224:
#             print('you should delete this guy:', image_path)
#
#
#
#         path = image_path.replace('.jpg', '.pth')
#
#         feat_path = os.path.join(feat_dir, class_name, path.split('/')[-1])
#
#         lock_path = feat_path + '.lock'
#         if os.path.exists(feat_path):
#             continue
#         if os.path.exists(lock_path):
#             continue
#         try:
#             os.makedirs(lock_path)
#         except:
#             continue
#
#         # get features
#         image = torch.unsqueeze(image, 0)
#         # print '\n', image.shape
#         feat = cnn(image)  # [1, 3, 224, 224]
#
#         if not os.path.exists(os.path.dirname(feat_path)):
#             try:
#                 os.makedirs(os.path.dirname(feat_path))
#                 print('## Make Directory: %s' % feat_path)
#             except:
#                 pass
#
#
#
#         torch.save(feat, feat_path)
#
#
#         if i % 100 == 0:
#             print('extracting feature [%d / %d]' % (i, len(images)))
#
#         cmd = 'rm -r %s' % lock_path
#         os.system(cmd)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn', default=os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'resnet50-base.pth'))
    parser.add_argument('--awa2_split', default=os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'awa2-split.json'))
    parser.add_argument('--awa2_path', default=os.path.join('/home/gyx/Data/Ori_DATA', 'AwA2', 'JPEGImages'))
    parser.add_argument('--feat_dir', default=os.path.join(DATA_DIR_PREFIX, 'AWA2', 'Test_DATA_feats_GPM'))

    parser.add_argument('--stage', default='test')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    # set_gpu(args.gpu)
    '''
    awa2-split.json:
    awa2_split[train], awa2_split[test], awa2_split[train_names], awa2_split[test_names]
    '''
    awa2_split_path = args.awa2_split
    awa2_split = json.load(open(awa2_split_path, 'r'))
    test_names = awa2_split['test_names']
    print test_names
    ## load resnet model
    cnn = resnet.make_resnet50_base()
    cnn.load_state_dict(torch.load(args.cnn))
    cnn.eval()


    # the directory of raw AWA2 data, with images
    awa2_path = args.awa2_path


    for i, name in enumerate(test_names, 1):

        print i, ' ', name, ' extracting features....',

        extract_feat(awa2_path, name, cnn, args.feat_dir, stage=args.stage)





