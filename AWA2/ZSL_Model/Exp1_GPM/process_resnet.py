import json
import torch
import os


'''
input: resnet50-raw.pth (resnet pre-train model), imagenet-split.json
get: `fc-weights.json` and `resnet50-base.pth`
function: retrive pre-train resnet without last layer and get training classes' supervision (fc-weights)
'''



DATA_DIR_PREFIX = '/home/gyx/Data/KG_SS_DATA/ZSL/'
# DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'
EXP_NAME = 'AWA2/Exp1_GPM'


p = torch.load(os.path.join(DATA_DIR_PREFIX, 'materials', 'resnet50-raw.pth'))
w = p['fc.weight'].data
b = p['fc.bias'].data

# resnet50 without fc layer weights and biases
p.pop('fc.weight')
p.pop('fc.bias')
torch.save(p, os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'resnet50-base.pth'))

# extract features of training images
v = torch.cat([w, b.unsqueeze(1)], dim=1).tolist()
wnids = json.load(open(os.path.join(DATA_DIR_PREFIX, 'AWA2/materials', 'imagenet-split.json'), 'r'))['train']
wnids = sorted(wnids)
obj = []
for i in range(len(wnids)):
    obj.append((wnids[i], v[i]))
json.dump(obj, open(os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'fc-weights.json'), 'w'))

