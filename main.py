# coding=utf-8
import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
from tensorboardX import SummaryWriter

import funs
from dataset import TSNDataSet      # dataset是一个py文件
from models import TSN              # models也是一个py文件
from transforms import *            # transforms是一个py文件
from opts import parser             # 此处读取命令行指令，也是一个py文件
import re

print('start time:', time.asctime(time.localtime(time.time())))
best_prec1 = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
weight_contrastive = 0.5

# 测试git提交

def main():
    global args, best_prec1         # args是接受命令行输入的全局变量  在这里定义层全局变量
    args = parser.parse_args()      # 获取命令行的输入执行

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    model = TSN(num_class, args.num_segments, args.modality,        # 构建神经网络模型  构造模型的时候num_segments值固定了，不能随意更改
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn, batch_size=args.batch_size)


    crop_size = model.crop_size             # 数据增强的参数这设置
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()       # 设置模型并行运算时的GPU块数，将模型设置为并行计算的方式

    if args.resume:             # 判断是从头开始训练还是从效果最好的神经网络结构开始训练
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # pretrained_dict = checkpoint['state_dict']
            # model_dict = model.state_dict()
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # model_dict.update(pretrained_dict)
            # model.load_state_dict(model_dict)
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
            print(("=> best prec1 '{}' "
                  .format(best_prec1)))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True
    word = 'base_model'
    # freeze(word, model)

    # Data loading code         选择数据的模式
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)       # GroupNormalize()是一个类
    else:
        normalize = IdentityTransform()     # 是一个类

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5
    #/ home / ange / projects / temporal - segment - works / data / ucf101_rgb_train_split_1.txt
    # 训练数据集的加载
    train_loader = torch.utils.data.DataLoader(
        TSNDataSet("",
                   args.train_list,
                   args,  # args不是包含所有的参数吗，直接出入args不就行了
                   num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])
                   ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # 验证集加载
    val_loader = torch.utils.data.DataLoader(
        TSNDataSet("",
                   args.val_list,
                   args,
                   num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   random_shift=False,
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
                  # batch_size=int(2*args.batch_size/15),
                  batch_size=args.batch_size,
                  shuffle=False,
                  num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer                                        定义使用的损失函数
    if args.loss_type == 'nll':     # nll最大似然 / log似然代价函数
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    criterion_contrastive = funs.ContrastiveLoss(margin=1)

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,                                                   # 设置优化方式
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:                                                                       # 选择是否进行评估验证
        validate(val_loader, model, criterion, 0)
        return

    #for epoch in range(args.start_epoch, args.epochs):                                      # 调整学习率
    with open('experiment_result.txt', 'a') as f:
        f.writelines('\n\n epoch: {0}  batch_size: {1}  init_lr: {2}  lr_step: {3}'.format(args.epochs, args.batch_size,args.lr, args.lr_steps))
        f.writelines('\n'+str(args))
    #
    # for epoch in range(args.start_epoch, args.epochs):
    if args.scratch:  # 决定是否从头开始训练 初始参数不变 只是学习率这个参数会变化
        args.start_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        if args.contrastive:  # 选择是否加入对比损失
            train_contrastive(train_loader, model, criterion, criterion_contrastive, optimizer, epoch)
        else:
            train(train_loader, model, criterion, optimizer, epoch)                             # 训练一个epoch
        print('epoch end time:', time.asctime(time.localtime(time.time())))
        # evaluate on validation set                                                        # 在验证集上评估训练效果
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))
            # remember best prec@1 and save checkpoint
            is_best = prec1 > (best_prec1)
            print('model is_best: ', is_best)
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, args.save_model_path)


# 定义训练函数
def train_contrastive(train_loader, model, criterion, criterion_contrastive, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):  # 读取数据的返回值是[[图像，标签]， [图像,标签]]
        # measure data loading time
        data_time.update(time.time() - end)

        target_0 = target[0].cuda(async=True)
        target_1 = target[1].cuda(async=True)
        input_var_0 = torch.autograd.Variable(input[0])  #[4,15,224,224]
        input_var_1 = torch.autograd.Variable(input[1])  #[4,15,224,224]
        contrastive_label = target_0 == target_1

        target_var_0 = torch.autograd.Variable(target_0)
        target_var_1 = torch.autograd.Variable(target_1)

        # compute output
        output_0 = model(input_var_0)
        output_1 = model(input_var_1)

        cro_loss = criterion(output_0, target_var_0)
        contrastive_loss = criterion_contrastive(output_0, output_1, contrastive_label)
        loss = cro_loss + weight_contrastive * contrastive_loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output_0.data, target_0, topk=(1,5))
        losses.update(loss.data[0], input_var_0.size(0))
        top1.update(prec1[0], input_var_0.size(0))
        top5.update(prec5[0], input_var_0.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))


    end_time = time.asctime(time.localtime(time.time()))
    with open('experiment_result.txt', 'a') as f:
        f.writelines("\n[epoch %d]: epoch loss = %f,Prec@1 = %f, Prec@5 = %f, lr=%f, end_time=%s, batch_size=%d" % (epoch + 1, losses.avg ,top1.avg,top5.avg,optimizer.param_groups[-1]['lr'],end_time,output_0.shape[1]))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)  #[4,15,224,224]
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))

    end_time = time.asctime(time.localtime(time.time()))
    with open('experiment_result.txt', 'a') as f:
        f.writelines("\n[epoch %d]: epoch loss = %f,Prec@1 = %f, Prec@5 = %f, lr=%f, end_time=%s, batch_size=%d" % (epoch + 1, losses.avg ,top1.avg,top5.avg,optimizer.param_groups[-1]['lr'],end_time,output.shape[1]))


# 定义验证函数
def validate(val_loader, model, criterion, iter, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # tmp = args.batch_size
    # args.batch_size = int(2*tmp/3)
    # print(args.batch_size)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            output = model(input_var)
            # output0 = output0.transpose(1, 2)
            loss = criterion(output, target_var)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1,5))

            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5)))
    end_time = time.asctime(time.localtime(time.time()))
    with SummaryWriter('/home/ange/project-2022/tsn-20200722/tsn-20200722-output/log') as writer:
        writer.add_scalar('val loss', losses.val, iter)
        writer.add_scalar('prec@1', top1.val, iter)
        writer.add_scalar('prec@5', top5.val, iter)
    # with open('experiment_result.txt', 'a') as f:
    #     f.writelines("\n[epoch %d]: epoch loss = %f,acc = %f, end_time=%f" % (i, losses.avg ,top1.avg,end_time))
    with open('experiment_result.txt', 'a') as f:
        f.writelines(('\nTest: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t' 'end_time {end_time}'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5, end_time=end_time)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses)))
    # args.batch_size = tmp
    return top1.avg


# 定义存储整个网络结构
def save_checkpoint(state, is_best, save_model_path, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_model_path):  # 如果保存模型的路径不存在则创建该文件夹
        os.makedirs(save_model_path)
    filename = save_model_path + '_'.join((args.snapshot_pref, args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = save_model_path + '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):     # 遇到类的定义会进入类的里面，遇到def直接跳过。
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 定义调整学习率的函数
def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))  # 0.1  0701
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


#　定义计算精度的函数
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # with open('experiment_result1.txt', 'a') as f:
    #     #f.writelines("\npredict = %0.3f, label = %0.3f " % (pred, correct))
    #     f.writelines("\npredict = %s" % (str(pred)))
    #     f.writelines("\nlabel11 = %s " % (str(target)))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def freeze(word,net):
    for name, p in net.named_parameters():
        str1 = name
        a = [m.start() for m in re.finditer(word, str1)]
        if a:
            p.requires_grad = False
        else:
            p.requires_grad = True
        if p.requires_grad:
            print(name)

def paras_load(pretrained_dict, model_dict):
    #  model is a model preparing to train. net has benn trained
    pretrained_dict = net.state_dict()
    #model_dict = model.state_dict()
    pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    #model.load_state_dict(model_dict)
    return model_dict

if __name__ == '__main__':
    main()      # 从这里开始进入主函数，一步一步执行
    print('end time:', time.asctime(time.localtime(time.time())))
