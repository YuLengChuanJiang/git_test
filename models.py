# coding=utf-8
from torch import nn

from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant
import time
import numpy
import ms_model
import torch
import  torch.nn.functional as F
from collections import OrderedDict
import math
import heapq
# count = 0

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 crop_num=1, partial_bn=True, batch_size=8):
        super(TSN, self).__init__()     # python2的一种写法，调用的是弗雷nn.module中的方法__init__()
        self.modality = modality        # RGB
        self.num_segments = num_segments        # 3
        self.reshape = True
        self.before_softmax = before_softmax    # true
        self.dropout = dropout
        self.crop_num = crop_num        # 1
        self.consensus_type = consensus_type    # 'avg'
        self.count = 0
        self.ms_stage = 1
        self.extract_features = []
        self.batch_size = batch_size
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5     # 如果modality==RGB,new_length设置为1否则设置为5
        else:
            self.new_length = new_length

        print(("""
                Initializing TSN with base model: {}.
                TSN Configurations:
                input_modality:     {}
                num_segments:       {}
                new_length:         {}
                consensus_module:   {}
                dropout_ratio:      {}
                """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model)

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)        # ??
        # feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        #self.BN = nn.BatchNorm1d(num_features=101)
        # self.ms_layer = ms_model.MultiStageModel(num_stages=self.ms_stage, num_layers=10, num_f_maps=64, dim=101, num_classes=101)


        # self.prepare_attention(self.num_segments, 101)
       # self.conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=self.num_segments, stride=1, padding=int((self.num_segments-1)/2))
        #self.conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1,padding=1)
# num_f_maps represent number of filter
        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn       # _开头的变量是受保护的变量
        if partial_bn:
            self.partialBN(True)

    def related_features(self,features, x, n):
        mostrelate = torch.zeros(features.shape[0], n,features.shape[2])
        for j in range(x.shape[0]):
            a = x[j,:]
            value3 = heapq.nlargest(n,a)
            value_pos = map(a.tolist().index, heapq.nlargest(n,a))
            tmp = list(value_pos)
            _,value_pos_seq = torch.sort(torch.tensor(tmp), dim=0)
            for i in range(len(_)):
                mostrelate[j,i,:]=features[j,_[i],:]
        return mostrelate



    def prepare_attention(self, num_segments, num_class):
        self.attention_layer = nn.Sequential(OrderedDict([
                            ('att_fc1', nn.Linear(num_segments*num_class, num_segments)),
                            ('att_tanh', nn.Tanh()),
                            ('att_fc2', nn.Linear(num_segments, num_segments)),
                            ('att_sfx', nn.Softmax())  # out:[]
                                                    ]))

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))       # 设置属性值，对象为self.base_model，属性为self.base_model.last_layer_name，设置的属性值为nn.Dropout(p=self.dropout)
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model:       # 如果resnet或者vgg在base_model中
            self.base_model = getattr(torchvision.models, base_model)(True)
            # getattr() 函数用于返回一个对象属性值。torchvision.model pytorch 自带的复现模型。这句话相当于 models.resnet101(pretrained=True)
            # print self.base_model
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]     # 重新设置输入的均值
                self.input_std = [np.mean(self.input_std)]      # 重新设置输入的方差
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'BNInception':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
                  
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]


    def forward(self, input):  # input [8,9,224,224]
        # global count
        # count +=1
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))  # base_out [36,2048]

        # extract_features=base_out # no memory out

        if self.dropout > 0:
            base_out = self.new_fc(base_out)  # base_out [36,101]

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])  # base_out=[12,3,101]

        output = self.consensus(base_out)

        return output.squeeze(1)

    def tsn_ms_att(self, base_out):
        mask = torch.ones(int(base_out.shape[0]), 101, int(base_out.shape[1]))
        tmp1 = base_out.transpose(2,1)
        output_ms = self.ms_layer(tmp1, mask.cuda())
        a = output_ms.view(output_ms.shape[1], -1)
        output_att = self.attention_layer(a)  # [batch_size, num_segments]
        output_ms = output_ms.squeeze(0)
        output_att1 = torch.mul(output_ms.transpose(0,1).reshape(output_ms.shape[1],-1),output_att.view(1,-1))  # [classess, batch_size*num_segments]
        output_att2 = output_att1.reshape(output_ms.transpose(0,1).shape).transpose(0,1).transpose(1, 2) # [4,303]  # [bz,ns,cl]
        output1 = torch.sum(output_att2,dim=1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
