# -*- coding: utf-8 -*-
# @Time : 2022/6/4 10:32
# @Author : 冷川江
# @Site : 
# @File : feature_cluster.py

import argparse
import sys
import numpy as np

sys.path.append('.')

weight = 1.5

from pyActionRecog.utils.video_funcs import default_aggregation_func
from pyActionRecog.utils.metrics import mean_class_accuracy, softmax


def calculate_recog_true_once(dis_r, dis_f, threshold1, threshold2, recog_fu, recog1_r, recog1_f):
    final = np.zeros(len(recog1_f))
    for i in range(len(recog1_f)):
        dis_sub = dis_r[i] - dis_f[i]
        if dis_sub < threshold1 and dis_sub > threshold2:
            final[i] = recog_fu[i]
        elif dis_sub < threshold1:
            final[i] = recog1_r[i]
        else:
            final[i] = recog1_f[i]

    recog_true = np.sum(final == np.array(label1_r))
    print('threshold1: {},  threshold2: {}, recong_true: {}'.format(threshold1, threshold2, recog_true))


def calculate_recog_similarity(feature_r, rgb_each_class_feature, recog1_r):
    dis_r = np.zeros(len(feature_r))
    for i in range(len(feature_r)):
        dis_r[i] = np.linalg.norm(feature_r[i] - rgb_each_class_feature[recog1_r[i]], axis=0, keepdims=True)[0]  # 只保存数据
    return dis_r


def read_action_txt(txt_path):
    with open(txt_path) as f:
        action = []
        while 1:
            line = f.readline()
            if not line:
                break
            action.append(line.strip().split(' ')[0])
    return action


def singe_stream(*args):  # 进阶版程序，根据输入参数进行不同执行不同的计算  第一个参数是rgb特征，第二个参数是flow特征，第三个参数是权重
    """
    对单流npy数据的处理，进行识别
    :param r_1_n_scores: 读取的npy文件
    :return: 识别的结果和对应的标签
    """
    recog1 = []
    label1 = []
    feature_return = []
    print('paras num', len(args))
    if len(args) == 1:
        for score_vec in args[0]:
            # agg_score_vec = [default_aggregation_func(x, normalization=False, crop_agg=getattr(np, 'mean')) for x in score_vec]
            feature = np.mean(score_vec[0].squeeze(), axis=0)
            feature_return.append(feature)
            recog1.append(np.argmax(softmax(feature)))
            label1.append(score_vec[1].item())
        return recog1, label1, feature_return
    if len(args) == 3:
        fusion_feature = []
        for i in range(len(args[0])):
            fusion_feature.append(np.mean(args[0][i][0].squeeze(), axis=0) + args[2] * np.mean(args[1][i][0].squeeze(), axis=0))
            recog1.append(np.argmax(softmax(np.mean(args[0][i][0].squeeze(), axis=0) + args[2] * np.mean(args[1][i][0].squeeze(), axis=0))))
            label1.append(args[0][i][1].item())
        return recog1, label1, fusion_feature
    return


def only_fuse_feature(*args):  # 进阶版程序，根据输入参数进行不同执行不同的计算  第一个参数是rgb特征，第二个参数是flow特征，第三个参数是权重
    """
    对单流npy数据的处理，进行识别
    :param r_1_n_scores: 读取的npy文件
    :return: 识别的结果和对应的标签
    """
    recog1 = []
    label1 = []
    print('paras num', len(args))
    if len(args) == 1:
        for score_vec in args[0]:
            # agg_score_vec = [default_aggregation_func(x, normalization=False, crop_agg=getattr(np, 'mean')) for x in score_vec]
            feature = np.mean(score_vec[0].squeeze(), axis=0)
            # recog1.append(np.argmax(softmax(feature)))
            label1.append(score_vec[1].item())
        return feature, label1
    if len(args) == 3:
        fusion_feature = []
        for i in range(len(args[0])):
            fusion_feature.append(np.mean(args[0][i][0].squeeze(), axis=0) + args[2] * np.mean(args[1][i][0].squeeze(), axis=0))
            # recog1.append(np.argmax(softmax(np.mean(args[0][i][0].squeeze(), axis=0) + args[2] * np.mean(args[1][i][0].squeeze(), axis=0))))
            label1.append(args[0][i][1].item())
        return fusion_feature, label1
    return


def adjust_data_and_dim(r_1_n_scores):
    r_1_n_new = []
    for i in r_1_n_scores:
        r_1_n_new.append(i[0].squeeze())
    return r_1_n_new


def classify_train_set(train_set, train_label):
    # 计算融合特征的时候已经进行了一次取均值
    train_class = [[] for i in range(101)]
    if len(train_set.shape) == 3:
        for i in range(101):  # 将标签为1-100的训练数据分别划分在一起
            pos = np.where(train_label == i)[0]

            for j in pos:
                    train_class[i].append(np.mean(train_set[j], axis=0))
    else:  # 如果训练集的形状为2 则不对特征取均值
        for i in range(101):  # 将标签为1-100的训练数据分别划分在一起
            pos = np.where(train_label == i)[0]
            for j in pos:
                train_class[i].append(train_set[j])

    return train_class

def  calculate_each_class_features(train_set, train_label):
    '''
    计算每一类视频的特征平均值
    :param train_set: 训练集所有视频的特征
    :param train_label: 训练集的标签
    :return: 101维的向量构成的列表  代表每类视频的特征均值
    '''
    train_calss_data = classify_train_set(train_set, train_label)
    # del train_set, train_label
    each_class_feature = []  # 每个动作类别的特征向量
    each_class_feature_similarity = []
    j = 0
    for i in train_calss_data:
        each_class_feature.append(np.mean(i, axis=0))
        mean_feature = each_class_feature[j]
        j += 1
        each_video_in_class_similarity = np.zeros(len(i))
        for k in range(len(i)):
            each_video_in_class_similarity[k] = np.linalg.norm(i[k] - mean_feature, axis=0, keepdims=True)[0]
        each_class_feature_similarity.append(each_video_in_class_similarity.mean())

    return each_class_feature, each_class_feature_similarity


def calculate_distance(feature_fu, rgb_each_class_feature, recog_fu):
    dis = []
    if len(feature_fu[0].shape) >1:
        for i in range(len(feature_fu)):
            distance = np.linalg.norm(np.mean(feature_fu[i], axis=0)-rgb_each_class_feature[recog_fu[i]] , axis=0, keepdims=True)[0]
            dis.append(distance)
    else:
        for i in range(len(feature_fu)):
            distance = np.linalg.norm(feature_fu[i] - rgb_each_class_feature[recog_fu[i]] , axis=0, keepdims=True)[0]
            dis.append(distance)
    return dis


def similarity_for_each_recog(each_class_feature, feature, recog):
    distance = np.linalg.norm(feature - each_class_feature[recog], axis=0, keepdims=True)[0]
    return distance


def recongnize_and_mean_similarity(recog1_f, rgb_each_class_feature_similarity):
    r_recog_slt = np.zeros(len(recog1_f))
    for i in range(len(recog1_f)):
        r_recog_slt[i] = rgb_each_class_feature_similarity[recog1_f[i]]
    return r_recog_slt


action1_path = '/home/ange/projects/temporal-segment-networks/data/ucf101_flow_val_split_1.txt'
action2_path = '/home/ange/projects/temporal-segment-networks/data/ucf101_flow_val_split_2.txt'
action3_path = '/home/ange/projects/temporal-segment-networks/data/ucf101_flow_val_split_3.txt'

r_1 = '/home/ange/project-2022/tsn-20200722/tsn-20200722-output/similar_features/resnet_rgb_score_seg5 _split1_220601.npz'
r_2 = '/home/ange/project-2022/tsn-20200722/tsn-20200722-output/similar_features/resnet_rgb_score_seg5_split2_220601.npz'
r_3 = '/home/ange/project-2022/tsn-20200722/tsn-20200722-output/similar_features/resnet_rgb_score_seg5_split3_220601.npz'

f_1 = '/home/ange/project-2022/tsn-20200722/tsn-20200722-output/similar_features/resnet_flow_score_seg5_split1_220604.npz'
f_2 = '/home/ange/project-2022/tsn-20200722/tsn-20200722-output/similar_features/resnet_flow_score_seg5_split2_220604.npz'
f_3 = '/home/ange/project-2022/tsn-20200722/tsn-20200722-output/similar_features/resnet_flow_score_seg5_split3_220604.npz'

r_1_n = np.load(r_1, allow_pickle=True)
r_2_n = np.load(r_2, allow_pickle=True)
r_3_n = np.load(r_3, allow_pickle=True)

f_1_n = np.load(f_1, allow_pickle=True)
f_2_n = np.load(f_2, allow_pickle=True)
f_3_n = np.load(f_3, allow_pickle=True)

r_1_n_scores = r_1_n['scores']
r_2_n_scores = r_2_n['scores']
r_3_n_scores = r_3_n['scores']

r_1_n_labels = r_1_n['labels']
r_2_n_labels = r_2_n['labels']
r_3_n_labels = r_3_n['labels']

f_1_n_scores = f_1_n['scores']
f_2_n_scores = f_2_n['scores']
f_3_n_scores = f_3_n['scores']

f_1_n_labels = f_1_n['labels']
f_2_n_labels = f_2_n['labels']
f_3_n_labels = f_3_n['labels']
del r_1_n, r_2_n, r_3_n
action1 = read_action_txt(action1_path)
action2 = read_action_txt(action2_path)
action3 = read_action_txt(action3_path)


r_1_n_new = adjust_data_and_dim(r_1_n_scores)
r_2_n_new = adjust_data_and_dim(r_2_n_scores)
r_3_n_new = adjust_data_and_dim(r_3_n_scores)
f_1_n_new = adjust_data_and_dim(f_1_n_scores)
f_2_n_new = adjust_data_and_dim(f_2_n_scores)
f_3_n_new = adjust_data_and_dim(f_3_n_scores)

feature_fu_2, label_fu_2 =  only_fuse_feature(r_2_n_scores, f_2_n_scores, 1.5)  # 融合特征
feature_fu_3, label_fu_3 =  only_fuse_feature(r_3_n_scores, f_3_n_scores, 1.5)  # 融合特征

train_set_rgb = np.concatenate((r_2_n_new, r_3_n_new))  # 这是训练集的rgb特征
train_set_flow = np.concatenate((f_2_n_new, f_3_n_new))
train_set_fusion = np.concatenate((feature_fu_2, feature_fu_3))

train_label = np.concatenate((r_2_n_labels, r_3_n_labels))  # 这是训练集的标签

# train_set_fusion = np.concatenate((weight * f_2_n_new + r_2_n_new, weight * f_3_n_new + r_3_n_new))
del r_2_n_new, r_3_n_new, r_2_n_labels, r_3_n_labels
# train_calss_data = classify_train_set(train_set, train_label)
# del train_set, train_label
# each_class_feature = []  # 每个动作类别的特征向量
# for i in train_calss_data:
#     each_class_feature.append(np.mean(i, axis=0))

rgb_each_class_feature, rgb_each_class_feature_similarity = calculate_each_class_features(train_set_rgb, train_label)
flow_each_class_feature, flow_each_class_feature_similarity = calculate_each_class_features(train_set_flow, train_label)
fusion_each_class_feature, fusion_each_class_feature_similarity = calculate_each_class_features(train_set_fusion, train_label)

# np.sum(np.array(recog1) == np.array(label1))  # 对识别结果正确与否进行统计
# 计算二范数： np.linalg.norm(x, axis=1, keepdims=True)
recog_fu, label_fu, feature_fu =  singe_stream(r_1_n_scores, f_1_n_scores, 1.5)  # 融合特征
# dis_all = [[] for i in range(len(feature_fu))]  # dis_all存储条数据与每类特征的相似度
# for i in range(len(feature_fu)):  # 这里是计算每个视频与训练集中每类视频的距离
#     dis = np.zeros(len(each_class_feature))
#     for j in range(len(each_class_feature)):
#         dis[j] = np.linalg.norm(feature_fu[i] - each_class_feature[j], axis=0, keepdims=True)[0]
#     dis_all[i].append(dis)


recog1_f, label1_f, feature_f = singe_stream(f_1_n_scores)
recog1_r, label1_r, feature_r = singe_stream(r_1_n_scores)

r_recong_slt = recongnize_and_mean_similarity(recog1_r, rgb_each_class_feature_similarity)
f_recong_slt = recongnize_and_mean_similarity(recog1_f, flow_each_class_feature_similarity)
fu_recong_slt = recongnize_and_mean_similarity(recog_fu, fusion_each_class_feature_similarity)

dis_r = calculate_recog_similarity(feature_r, rgb_each_class_feature, recog1_r)
dis_f = calculate_recog_similarity(feature_f, flow_each_class_feature, recog1_f)
dis_fu = calculate_recog_similarity(feature_fu, fusion_each_class_feature, recog_fu)

dis_all = np.vstack((r_recong_slt, dis_r, recog1_r, f_recong_slt, dis_f, recog1_f, fu_recong_slt, dis_fu, recog_fu, label_fu)).T



# 已知量：识别结果、视频特征、每一类聚合特征
dis = []
for i in feature_fu:
    distance = np.linalg.norm(np.mean(feature_fu[0], axis=0)-rgb_each_class_feature[0] , axis=0, keepdims=True)[0]
    dis.append()


threshold1 = 10
threshold2 = 20
final = np.zeros(len(recog1_f))
for i in range(len(recog1_f)):
    dis_sub = dis_r - dis_f
    if dis_sub < threshold1 and dis_sub > threshold2:
        final[i] = recog_fu[i]
    elif dis_sub < threshold1:
        final[i] = recog1_r[i]
    else:
        final[i] = recog1_f[i]

print(np.sum(final == np.array(label1_r)))


# np.sum(np.array(recog_fu) == np.array(label1_r))
# dis_r = calculate_recog_similarity(feature_r, rgb_each_class_feature, recog1_r)
# dis_f = calculate_recog_similarity(feature_f, flow_each_class_feature, recog1_f)
# dis_fu = calculate_recog_similarity(feature_fu, fusion_each_class_feature, recog_fu)
# dis_all = np.vstack((dis_r, recog1_r, dis_f, recog1_f, dis_fu, recog_fu, label_fu)).T

#
# for i in range(recog_wrong_index.shape[0]):
#     name = action1[recog_wrong_index[i]]
#     lab = label_fu[recog_wrong_index[i]]
#     lab_w = recog_fu[recog_wrong_index[i]]
#     act = action_label[str(lab_w)]
#     print('video name {}, true label {}, wrong recognition label {}, wrong action name {}'.format(name, lab, lab_w, act))

# 将错误的视频单独复制出来
# import shutil
# for i in range(recog_wrong_index.shape[0]):
#     remove_video = action1[recog_wrong_index[i]]
#     lab_w = recog_fu[recog_wrong_index[i]]
#     act = action_label[str(lab_w)]
#     video_name = remove_video.split('/')[-1]
#     action_name = video_name.split('_')[1]
#     vid_path = remove_video.replace('images', 'UCF-101/' + action_name) + '.avi'
#     target_path = '/home/ange/project-2022/tsn-20200722/wrong_videos/' + video_name + '_wrong_' + act + '.avi'
#     shutil.copy(vid_path, target_path)






