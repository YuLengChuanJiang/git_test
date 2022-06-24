# coding=utf-8
import argparse     # 不是--开头的是必选参数，那他是怎样寻找图像帧的位置的？
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")     # 生成类的对象
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics'])      # 增加数据集的选项
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])           # 输入模态的选择
parser.add_argument('train_list', type=str)                                             # 训练集列表
parser.add_argument('val_list', type=str)                                               # 测试集列表

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="resnet101")                  # 使用的是基本的网络结构default="resnet101"
parser.add_argument('--num_segments', type=int, default=5)                      # 设置分段的数量 default=3
parser.add_argument('--consensus_type', type=str, default='avg',                # 得出一致性结果的方式
                    choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn'])
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.9, type=float,               # 设置随机失活率
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])

parser.add_argument('--scratch', type=bool, default=False)

# ========================= Learning Configs ==========================
# parser.add_argument('--epochs', default=340, type=int, metavar='N',              # epoch参数设置45
#                     help='number of total epochs to run')
# parser.add_argument('-b', '--batch-size', default=24, type=int,                # 设置batch-size的大v小256
#                     metavar='N', help='mini-batch size (default: 256)')
#
# parser.add_argument('--lr_steps', default=[15, 120, 260], type=float, nargs="+",      # 设置学习学习率衰减[20 40]
#                     metavar='LRSteps', help='epochs to decay learning rate by 10')


parser.add_argument('--epochs', default=80, type=int, metavar='N',              # epoch????45
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=20, type=int,                # ??batch-size??v?256
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+",      # [20 40]
                    metavar='LRSteps', help='epochs to decay learning rate by 10')

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,       # 设置学习率0.001  初始学习率
                    metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.5, type=float, metavar='M',         # 设置momentum
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,         # 设置权重衰减5e-4

                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')      # 设置输出的频率
parser.add_argument('--eval-freq', '-ef', default=1, type=int,   # 5
                    metavar='N', help='evaluation frequency (default: 5)')

#/home/ange/projects/tsn-ms-pytorch/tsn-pytorch20200701/_rgb_model_best.pth.tar
#/home/ange/projects/tsn-ms-pytorch/tsn-pytorch20200716/ucf101_res_rgb_0717_rgb_model_best.pth.tar
# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',        # 数据加载调用的进程
                    help='number of data loading workers (default: 4)')
# parser.add_argument('--resume', default='/home/ange/project-2022/tsn-20200722/tsn-20200722-output/source_model/ucf101_seg5_0723_rgb_model_best.pth.tar', type=str, metavar='PATH',           # 设置从最好的实验结果加载模型的位置
#                     help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',           # 设置从最好的实验结果加载模型的位置
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', default=False, dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="ucf101_seg5_0723")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',          # 手动设置开始的epoch
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)                # 设置GPU的数量default=None
parser.add_argument('--flow_prefix', default="", type=str)

parser.add_argument('--save_model_path', default="/home/ange/project-2022/tsn-20200722/tsn-20200722-output/models", type=str)


# 控制使用对比损失
parser.add_argument('--contrastive', default=False, type=bool, help='whether add a contrastive loss')






