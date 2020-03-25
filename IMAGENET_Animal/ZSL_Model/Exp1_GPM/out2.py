# coding=gbk
# -*- coding: utf-8 -*-

from __future__ import print_function

''' 
test the per class accuracy of original model and attentive model
'''
import sys
import os
import numpy as np
sys.path.append('../../')




# DATA_DIR_PREFIX = '/home/gyx/Data/KG_SS_DATA/ZSL/'
DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'
EXP_NAME = 'IMAGENET_Animal/Exp1_GPM/Exp_Test'



gcn_res_file = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'gcn_per_class_acc2.txt')
agcn_res_file = os.path.join(DATA_DIR_PREFIX, EXP_NAME, 'agcn_per_class_acc_weights1.txt')

# save high classification result
# we_file = os.path.join(DATA_DIR, 'Exp2_Per_Class_Acc', 'Explanation_Wname.txt')


def readResultFile(file_name):
    file_object = open(file_name, 'rU')
    result = dict()  # wnid - wn text
    try:
        for line in file_object:
            line = line[:-1]  # 直接 输出的 line 带"\n"，去除最后的 "\n"
            # print(line)
            res = line.split('\t')
            res_list = list()
            for i in range(1, len(res)):
                res_list.append(res[i])
            result[res[0]] = res_list

    finally:
        file_object.close()
    return result


gcn_res = readResultFile(gcn_res_file)
agcn_res = readResultFile(agcn_res_file)


# compare1: accuracy outperforme: hit@1, hit@2 agcn均高于gcn

agcn_out = 0
gcn_out = 0


all = 0

num = 5  # cnt_valid

cnt_gcn = 0
cnt_agcn = 0
hit_gcn = np.zeros(5)
hit_agcn = np.zeros(5)
# seen = list()
for name, result in gcn_res.items():
    weights = agcn_res[name][6]
    # weights_list = weights.split('|')  #
    # if weights.count('- seen -') == 1:
    if weights.count('|') > 2 and weights.count('- seen -') >= 1:
        all = all + 1
        cnt_gcn += int(gcn_res[name][num])
        cnt_agcn += int(agcn_res[name][num])
        for k in range(5):
            hit_gcn[k] += float(gcn_res[name][k])
            hit_agcn[k] += float(agcn_res[name][k])

        # for wei in weights_list:
        #     if '- seen -' in wei:
        #         seen_name = wei[8: wei.rindex('-')]
        #         # print("----", seen_name)
        #         seen.append(seen_name)

# print(len(set(seen)))
print(all)
print(cnt_gcn)
print(cnt_agcn)
test_acc_per_gcn = hit_gcn * 1.0 / cnt_gcn
acc_per_gcn = ['{:.2f}'.format(k * 100) for k in test_acc_per_gcn]
print("---gcn--", "-PER ACC- ", acc_per_gcn)


test_acc_per_agcn = hit_agcn * 1.0 / cnt_agcn
acc_per_agcn = ['{:.2f}'.format(k * 100) for k in test_acc_per_agcn]
print("---agcn--", "-PER ACC- ", acc_per_agcn)






# we_wname = list()
# with open(we_file, 'r') as fp:
#     for line in fp.readlines():
#         line = line[:-1]
#         we_wname.append(line)
#
# print len(we_wname)
#
# hc_we = 0
# for hc in atten_list:
#     if hc in we_wname:
#         hc_we = hc_we + 1
# print ("hc_we:", hc_we)




