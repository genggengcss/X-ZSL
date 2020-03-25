import os


Mat_DATA_DIR = '/home/gyx/X_ZSL/data/materials'
DATA_DIR = '/home/gyx/X_ZSL/data/GCNZ'
DATASET = 'ImNet_A'
EXP_NAME = 'Exp1_GCN'





unseen_file = os.path.join(DATA_DIR, DATASET, 'unseen.txt')

def obtain_wnid(wnid_file):
    animal = []
    nodes = open(wnid_file, 'rU')
    try:
        for line in nodes:
            line = line[:-1]
            animal.append(line)
    finally:
        nodes.close()
    return animal


unseen_list = obtain_wnid(unseen_file)


# Target_DIR = os.path.join(DATA_DIR, DATASET, 'Test_DATA_feats')
# Source_DIR = '/home/gyx/Data/KG_SS_DATA/ZSL/IMAGENET/Test_DATA_feats'
#
# for unseen in unseen_list:
#     file = os.path.join(Source_DIR, unseen)
#     cmd = 'cp -r ' + file + ' ' + Target_DIR
#     os.system(cmd)


Target_DIR = '/home/gyx/X_ZSL/data/images/ImNet_A'
Source_DIR = '/home/gyx/Data/Ori_DATA/ImageNet_2_hops_Images'

for unseen in unseen_list:
    file = os.path.join(Source_DIR, unseen)
    cmd = 'cp -r ' + file + ' ' + Target_DIR
    os.system(cmd)