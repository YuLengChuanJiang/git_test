# coding=utf-8
import argparse
import sys
import numpy as np
sys.path.append('.')

from pyActionRecog.utils.video_funcs import default_aggregation_func
from pyActionRecog.utils.metrics import mean_class_accuracy

parser = argparse.ArgumentParser()
parser.add_argument('score_files', nargs='+', type=str)
parser.add_argument('--score_weights', nargs='+', type=float, default=None)
parser.add_argument('--crop_agg', type=str, choices=['max', 'mean'], default='mean')
args = parser.parse_args()
# 将两个得分文件循环读取放于列表中
score_npz_files = [np.load(x) for x in args.score_files]

if args.score_weights is None:
    score_weights = [1] * len(score_npz_files)
else:
    score_weights = args.score_weights
    if len(score_weights) != len(score_npz_files):
        raise ValueError("Only {} weight specifed for a total of {} score files"
                         .format(len(score_weights), len(score_npz_files)))
# 里面是字典形式的数据 scores对应的键值得第二列不需要
score_list = [x['scores'][:, 0] for x in score_npz_files]  # 对预测结果的提取，仍然是以列表的形式保存
label_list = [x['labels'] for x in score_npz_files]  # 对标签的提取

# label verification

# score_aggregation
agg_score_list = []
for score_vec in score_list:
    # 一个视频25帧取均值 agg_score_vec [3783,101]
    agg_score_vec = [default_aggregation_func(x, normalization=False, crop_agg=getattr(np, args.crop_agg)) for x in score_vec]
    # 将求均值后的数组放在列表中
    agg_score_list.append(np.array(agg_score_vec))  # np.array(agg_score_vec)将列表形式的数据转为数组[3783,101]

final_scores = np.zeros_like(agg_score_list[0])  # 生成存放每个视频最终融合后的向量的数组
for i, agg_score in enumerate(agg_score_list):
    # 将RGB中3783个视频的得分乘权重与Flow中3783个视频得分权重相乘相加  得到最后的得分
    final_scores += agg_score * score_weights[i]

# accuracy
acc = mean_class_accuracy(final_scores, label_list[0])
print 'Final accuracy {:02f}%'.format(acc * 100)