# coding=utf-8
# ！E:\python
# -*- coding: utf-8 -*-
# @Time : 2022/1/14 11:33
# @Author : 冷川江
# @Site : 
# @File : funs.py
import numpy as np
from torch import nn
import torch.nn.functional as F

# with open('recorder.txt', 'r') as f:
#     lines = f.readlines()
#     line = lines[-1].split(' ')[1].split('_')[1]


def read_txt_info(list_file, test_mode):
    '''
    这个函数的作用是根据训练集或者测试集的获取视频信息，并找到对比的视频信息
    :param list_file:  训练集txt形式的信息  表明视频的路径 标签 帧数等
    :param test_mode:   逻辑值，是否是测试模态
    :return:new_vid_info  提取上述信息并加入对比数据 形成的列表

    '''
    with open(list_file) as f:
        info = f.readlines()
    vid_info = [[] for i in range(len(info))]
    for i in range(len(info)):
        vid_info[i].extend(info[i].strip().split(' '))
    enhance_label = {2: 67, 16: 17, 19: 77, 22: 28, 23: 22, 28: 84, 33: 12, 37: 36, 44: 84, 60: 65, 70: 16, 80: 79, 98: 90}
    vid_pairs = 2
    if test_mode:  # 如果是测试模型直接返回单个数据形式的视频信息，否则继续执行
        return vid_info
    new_vid_info = []
    for vid in vid_info:
        second_vid_list = []
        if int(vid[2]) in enhance_label.keys():
            for second_vid in vid_info:
                if int(second_vid[2]) == enhance_label[int(int(vid[2]))]:
                    second_vid_list.append(second_vid)
        else:
            second_vid_list.append(vid_info[np.random.randint(0, len(vid_info)-1)])
        for i in range(vid_pairs):
            new_vid_info.append([vid, second_vid_list[np.random.randint(0, len(second_vid_list))]])

    return new_vid_info


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss  样本属于同一类别则标签为1否则标签为0
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9


    def forward(self, output1, output2, target, size_average=True):

        distances = (output2 - output1).pow(2).sum(1)  # squared distances

        losses = 0.5 * (target.float() * distances +
                        (1.0 + -1.0 * target.float()) * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))

        # 当视频类别不同时，target的值为0，target.float() * distances的值为0，起作用的是self.margin - (distances + self.eps).sqrt()；如果distance的值大于
        # self.margin,根据relu函数，此时损失值为0，如果小于self.margin，此时distance的值越大，loss值越小。loss项迫使distance变大，即动作之间的差异变大。
        # 当两个视频类别相同时，target的值为1，1.0 + -1.0 * target.float()为0，此时只剩target.float() * distances；那么distance越大，损失值越大。优化器让损失变小，
        # 相当于distance的值就会变小。该损失函数让相同类别的视频特征更为相似。
        return losses.mean() if size_average else losses.sum()
