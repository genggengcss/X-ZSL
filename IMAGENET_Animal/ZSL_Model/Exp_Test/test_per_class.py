# coding=gbk
# -*- coding: utf-8 -*-

from __future__ import print_function

''' 
test accuracy of single class (any model)
'''
import sys
sys.path.append('../../')
from ZSL.IMAGENET_Animal.ZSL_Model.Exp_Test.utils import *



# DATA_DIR_PREFIX = '/Users/geng/Data/Human_X_ZSL_DATA/'
DATA_DIR_PREFIX = '/Users/geng/Data/KG_SS_DATA/ZSL/'
EXP_DATA = 'IMAGENET_Animal/Exp1_GCN'
EXP_NAME = 'IMAGENET_Animal/Exp2_AGCN'


def test_imagenet_zero(weight_pred_file, has_train=False):

    test_feat_file_path_all = []
    testlabels_all = []
    test_feat_file_path_per = []
    testlabels_per = []

    pre_lbl = 398
    with open(Testlist_File) as fp:  # test_image_list.txt  测试数据文件
        lines = fp.readlines()
        for i in range(len(lines)):
            fname, lbl = lines[i].split()  # n03236735/n03236735_4047.JPEG 398

            assert int(lbl) >= 398
            # feat_name = os.path.join(feat_folder, fname.replace('.JPEG', '.mat'))
            feat_name = os.path.join(Test_Img_Feat_Folder, fname.replace('.JPEG', '.npz'))  # 获取对应图片的特征文件

            if not os.path.exists(feat_name):
                print('not feature', feat_name)
                continue


            if int(lbl) == pre_lbl:
                test_feat_file_path_per.append(feat_name)
                testlabels_per.append(int(lbl))

            if int(lbl) != pre_lbl or (i+1) == len(lines):
                test_feat_file_path_all.append(test_feat_file_path_per)
                testlabels_all.append(testlabels_per)
                wnid_test_index.append(pre_name.split('/')[0])

                test_feat_file_path_per = []  # test feature file: n03236735/n03236735_4047.npz
                testlabels_per = []  # test class label: 398
                test_feat_file_path_per.append(feat_name)
                testlabels_per.append(int(lbl))

            pre_lbl = int(lbl)
            pre_name = fname


    with open(Classids_File_Retrain) as fp:  # corresp-awa.json
        classids = json.load(fp)
    with open(Word2vec_File, 'rb') as fp:  # glove
        # word2vec_feat = pkl.load(fp, encoding='iso-8859-1')
        word2vec_feat = pkl.load(fp)

    # obtain training results
    with open(weight_pred_file, 'rb') as fp:  #
        weight_pred = pkl.load(fp)
    weight_pred = np.array(weight_pred)

    print('weight_pred output', weight_pred.shape)


    # process 'train' classes. they are possible candidates during inference
    invalid_wv = 0  # count the number of invalid class embedding
    labels_testval, word2vec_testval = [], []  # zsl: unseen label and its class embedding
    weight_pred_testval = []  # zsl: unseen output feature
    for j in range(len(classids)):
        t_wpt = weight_pred[j]
        if has_train:
            if classids[j][0] < 0:
                continue
        else:
            if classids[j][1] == 0:
                continue

        if classids[j][0] >= 0:
            t_wv = word2vec_feat[j]
            if np.linalg.norm(t_wv) == 0:  # 求范数
                invalid_wv = invalid_wv + 1
                continue
            labels_testval.append(classids[j][0])
            word2vec_testval.append(t_wv)

            feat_len = len(t_wpt)
            t_wpt = t_wpt[feat_len - feat_dim: feat_len]
            weight_pred_testval.append(t_wpt)
    weight_pred_testval = np.array(weight_pred_testval)
    print('skip candidate class due to no word embedding: %d / %d:' % (invalid_wv, len(labels_testval) + invalid_wv))
    print('candidate class shape: ', weight_pred_testval.shape)

    weight_pred_testval = weight_pred_testval.T
    labels_testval = np.array(labels_testval)
    print('final test classes: ', len(labels_testval))

    # remove invalid unseen classes(wv = 0)
    valid_class = np.zeros(22000)
    invalid_unseen_wv = 0
    for j in range(len(classids)):
        if classids[j][1] == 1:  # unseen classes
            t_wv = word2vec_feat[j]
            t_wv = t_wv / (np.linalg.norm(t_wv) + 1e-6)

            if np.linalg.norm(t_wv) == 0:
                invalid_unseen_wv = invalid_unseen_wv + 1
                continue
            valid_class[classids[j][0]] = 1


    ## imagenet 2-hops topK result
    topKs = [1]
    top_retrv = [1, 2, 3, 5, 10, 20]
    hit_count_all = np.zeros((len(topKs), len(top_retrv)))
    cnt_valid_all = 0  # count test images
    acc_per_dict = dict()

    for i in range(len(testlabels_all)):

        hit_count_per = np.zeros((len(topKs), len(top_retrv)))
        cnt_valid_per = 0

        for j in range(len(test_feat_file_path_all[i])):
            test_feat_file = test_feat_file_path_all[i][j]
            if valid_class[testlabels_all[i][j]] == 0:   # remove invalid unseen classes
                break
            cnt_valid_per = cnt_valid_per + 1
            cnt_valid_all = cnt_valid_all + 1

            test_feat = np.load(test_feat_file)
            test_feat = test_feat['feat']

            scores = np.dot(test_feat, weight_pred_testval).squeeze()

            scores = scores - scores.max()
            scores = np.exp(scores)
            scores = scores / scores.sum()

            ids = np.argsort(-scores)

            for top in range(len(topKs)):
                for k in range(len(top_retrv)):
                    current_len = top_retrv[k]

                    for sort_id in range(current_len):
                        lbl = labels_testval[ids[sort_id]]
                        if int(lbl) == testlabels_all[i][j]:
                            hit_count_per[top][k] = hit_count_per[top][k] + 1
                            hit_count_all[top][k] = hit_count_all[top][k] + 1
                            break


        if cnt_valid_per != 0:
            print('processing %d / %d ' % (i+1, len(testlabels_all)))
            wn_name = WNID_WNAME[wnid_test_index[i]]
            test_acc_per = hit_count_per * 1.0 / cnt_valid_per
            acc_per = ['{:.2f}'.format(i * 100) for i in test_acc_per[0]]
            print("CLASS-", wn_name, "-PER ACC-", acc_per)
            acc_per_dict[wn_name] = acc_per

    per_class_acc = os.path.join(DATA_DIR_PREFIX, 'EXP_NAME', 'per_class_acc_attention.txt')
    wr_fp = open(per_class_acc, 'w')
    for name, result in acc_per_dict.items():
        line = name
        for res in result:
            line = line + "\t" + res
        wr_fp.write('%s\n' % line)
    wr_fp.close()

    hit_count_all = hit_count_all * 1.0 / cnt_valid_all
    # print(hit_count)
    print('total: %d', cnt_valid_all)
    return hit_count_all



wnid_test_index = []


if __name__ == '__main__':



    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--exp', type=str, default='Exp2_AGCN', help='choice: [Exp1_GCN,Exp2_AGCN]')
    parser.add_argument('--featname', type=str, default='1200', help='choice: [500, 1200, 1500]')
    args = parser.parse_args()

    training_outputs = os.path.join(DATA_DIR_PREFIX, args.exp, 'glove_res50/output/feat_') + args.featname



    print('\nEvaluating ...\nPlease be patient for it takes a few minutes...')

    res = test_imagenet_zero(weight_pred_file=training_outputs)

    output = ['{:.2f}'.format(i * 100) for i in res[0]]


    print('----------------------')
    print('model : ', training_outputs)
    print('result: ', output)
    print('----------------------')


