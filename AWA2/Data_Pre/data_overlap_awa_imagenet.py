# coding=gbk
# -*- coding: utf-8 -*-

# instruction:
# the common item between awa and imagenet 2010 1k:
# training set:24/40; testing set:0/10


# imagenet 2012 1k
DATA_FILE_2012_1k = '/Users/geng/Data/KG_SS_DATA/ZSL/IMAGENET/materials/seen_2012_1k.txt'


train_list = ["n02071294", "n02363005", "n02110341", "n02123394", "n02106662", "n02123597", "n02445715", "n01889520", "n02129604", "n02398521", "n02128385", "n02493793", "n02503517", "n02480855", "n02403003", "n02481823", "n02342885", "n02118333", "n02355227",  "n02324045", "n02114100", "n02085620", "n02441942", "n02444819", "n02410702", "n02391049", "n02510455", "n02395406", "n02129165", "n02134084", "n02106030", "n02403454", "n02430045", "n02330245", "n02065726", "n02419796", "n02132580", "n02391994", "n02508021", "n02432983"]
test_list = ["n02411705", "n02068974", "n02139199", "n02076196", "n02064816", "n02331046", "n02374451", "n02081571", "n02439033", "n02127482"]

print(len(train_list))

wnids_2012 = list()
with open(DATA_FILE_2012_1k, 'r') as fp:
    for line in fp.readlines():
        wnid = line.strip()
        wnids_2012.append(wnid)


awa_in_2012 = list()



for wnid in train_list:
    if wnid in wnids_2012:
        awa_in_2012.append(wnid)



print(len(awa_in_2012))






