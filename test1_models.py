# coding=utf-8
import argparse
import time
import torch

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from dataset import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule

import torch.backends.cudnn as cudnn
import torch._utils
import os
import sys

start_time = time.asctime(time.localtime(time.time()))
print('start time:', start_time)

# 跑测试的时候好像是需要4块卡

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# optionscv
parser = argparse.ArgumentParser(description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)  #25
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.9)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')  # 4
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')


args = parser.parse_args()
exe_file = os.path.abspath(sys.argv[0])
#print(args)
with open('experiment_result.txt', 'a') as f:
    f.writelines('\n\n' + str(args))
    f.writelines('\n' + exe_file)



if args.dataset == 'ucf101':
    num_class = 101
elif args.dataset == 'hmdb51':
    num_class = 51
elif args.dataset == 'kinetics':
    num_class = 400
else:
    raise ValueError('Unknown dataset '+args.dataset)

net = TSN(num_class, 1, args.modality,  # mo ren di 2ge canshu wei 1
          base_model=args.arch,
          consensus_type=args.crop_fusion_type,
          dropout=args.dropout)
# net = TSN(num_class, 5, args.modality,  # mo ren di 2ge canshu wei 1
#           base_model=args.arch,
#           consensus_type=args.crop_fusion_type,
#           dropout=args.dropout)


checkpoint = torch.load(args.weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

# net = torch.nn.DataParallel(net, device_ids=args.gpus).cuda()
# cudnn.benchmark = True

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}

# for name in base_dict:
#     # tmp = name[:20]
#     # size1 = len(base_dict[name].size())
#     if name[:20] == 'base_model.inception':
#         if len(base_dict[name].size()) == 1:
#             base_dict[name] = base_dict[name].unsqueeze(0)

net.load_state_dict(base_dict)

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

data_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.test_list, num_segments=args.test_segments,
                   new_length=1 if args.modality == "RGB" else 5,
                   modality=args.modality,
                   image_tmpl="img_{:05d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
        batch_size=1, shuffle=False,  # batch_size =1表明一次只读取一个视频数据
        num_workers=args.workers * 2, pin_memory=True)

# if args.gpus is not None:
#     devices = [args.gpus[i] for i in range(args.workers)]
# else:
#     devices = list(range(args.workers))
devices = args.gpus

#net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
# net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
net = torch.nn.DataParallel(net, device_ids=devices).cuda()
net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
output = []
features=[]

def eval_video(video_data):
    i, data, label = video_data
    num_crop = args.test_crops
    temp2 = 0  # 19
    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)
    with torch.no_grad():
        input_var = torch.autograd.Variable(data.view(-1, 5*length, data.size(2), data.size(3)),  #[250,3,224,224]
                                    volatile=True)

#    temp1,temp2 = net(input_var)  # 19
        temp1= net(input_var)  # 19
        rst0 = temp1.data.cpu().numpy().copy()  # 19
        #rst = rst0[:,3,:]    # 712
        rst = rst0

    return i, rst.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape(
        (args.test_segments, 1, num_class)
    ), label[0]
    # return i, rst.reshape((num_crop, 5, num_class)).mean(axis=0).reshape(
    #     (5, 1, num_class)
    # ), label[0]


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)



for i, (data, label) in data_gen:  # data_gen是通过dataloader加载进来的数据 对所有的数据按batch进行遍历
    if i >= max_num:  # 结束条件是迭代次数大于视频的个数
        break
    rst = eval_video((i, data, label))  # 调用模型进行验证 输出值包含两个量 第二个是输出的特征
    output.append(rst[1:])



    cnt_time = time.time() - proc_start_time
    print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                    total_num,
                                                                    float(cnt_time) / (i+1)))


video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]

video_labels = [x[1] for x in output]

cf = confusion_matrix(video_labels, video_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt

print(cls_acc)

print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))

if args.save_scores is not None:

    # reorder before saving
    name_list = [x.strip().split()[0] for x in open(args.test_list)]

    order_dict = {e:i for i, e in enumerate(sorted(name_list))}

    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)

    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]

    np.savez(args.save_scores, scores=reorder_output, labels=reorder_label)


end_time = time.asctime(time.localtime(time.time()))
print('end time:', end_time)
with open('experiment_result.txt', 'a') as f:
    f.writelines('\n')
    f.writelines('\n')
    f.writelines('start_time:'+start_time+'\n')
    f.writelines('end_time:' + end_time + '\n')
    f.writelines('acc:'+str(np.mean(cls_acc) * 100)+'\n')
    f.writelines(str(args))

