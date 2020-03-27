from __future__ import print_function

import sys
import os
import numpy as np
import json
'''
process unseen class with their impressive seen classes
'''

from collections import Counter





agcn_res_file = '../data/X_ZSL/example/agcn_seen_class1.txt'



def readImNet(file_name):
    file_object = open(file_name, 'rU')
    result = dict()  # unseen: [seen list]
    try:
        for line in file_object:
            line = line[:-1]
            print(line)
            res = line.split('\t')
            if len(res) > 1:
                res_list = list()
                for i in range(1, len(res)):
                    res_list.append(res[i])
                result[res[0]] = res_list


    finally:
        file_object.close()
    return result



agcn_res = readImNet(agcn_res_file)
print(len(agcn_res))




output_file = '../data/X_ZSL/example/IMSC.json'
json.dump(agcn_res, open(output_file, 'w'))




# counts = list()
# for unseen, seens in agcn_res.items():
#     counts.append(len(seens))
#
#
#
#
# c = Counter(counts)
# print(c)