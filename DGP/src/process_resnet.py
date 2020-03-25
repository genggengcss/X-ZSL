import json
import torch
import os


'''
input: resnet50-raw.pth (resnet pre-train model), imagenet-split.json
get: `fc-weights.json` and `resnet50-base.pth`
function: retrive pre-train resnet without last layer and get training classes' supervision (fc-weights)
'''



data_root = '/home/gyx/X_ZSL/data'
data_dir = 'DGP'



DATA_DIR = os.path.join(data_root, data_dir)


p = torch.load(os.path.join(DATA_DIR, 'materials', 'resnet50-raw.pth'))
w = p['fc.weight'].data
b = p['fc.bias'].data
print w.shape
print b.shape
# resnet50 without fc layer weights and biases
p.pop('fc.weight')
p.pop('fc.bias')
torch.save(p, os.path.join(DATA_DIR, 'materials', 'resnet50-base.pth'))

# extract features of training images
v = torch.cat([w, b.unsqueeze(1)], dim=1).tolist()
wnids = json.load(open(os.path.join(DATA_DIR, 'materials', 'imagenet-split-animal.json'), 'r'))['train']
wnids = sorted(wnids)
obj = []
for i in range(len(wnids)):
    obj.append((wnids[i], v[i]))

# print len(obj[0][1])
json.dump(obj, open(os.path.join(DATA_DIR, 'materials', 'fc-weights.json'), 'w'))

