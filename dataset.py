# coding=utf-8
import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import torch

import funs


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property                   # 将类的方法调用变为属性调用，最终目的还是对属性进行操作，只不过在对属性进行操作时，通过方法对属性进行一定的限制
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoRecord_Contrastive(object):
    def __init__(self, row):
        self._data = row
        # self._data1 = row[0]

    @property                   # 将类的方法调用变为属性调用，最终目的还是对属性进行操作，只不过在对属性进行操作时，通过方法对属性进行一定的限制
    def path(self):
        return self._data[0][0]

    @property
    def num_frames(self):
        return int(self._data[0][1])

    @property
    def label(self):
        return int(self._data[0][2])

    @property
    def path2(self):
        return self._data[1][0]

    @property
    def num_frames2(self):
        return int(self._data[1][1])

    @property
    def label2(self):
        return int(self._data[1][2])

# class TSNDataSet(data.Dataset):
#     def __init__(self, root_path, list_file,
#                  num_segments=3, new_length=1, modality='RGB',
#                  image_tmpl='img_{:05d}.jpg', transform=None,
#                  force_grayscale=False, random_shift=True, test_mode=False):
#
#         self.root_path = root_path
#         self.list_file = list_file
#         self.num_segments = num_segments
#         self.new_length = new_length
#         self.modality = modality
#         self.image_tmpl = image_tmpl
#         self.transform = transform
#         self.random_shift = random_shift
#         self.test_mode = test_mode
#
#         if self.modality == 'RGBDiff':
#             self.new_length += 1        # Diff needs one more image to calculate diff
#
#         # self._parse_list()  # 此处是调用正常处理TSN数据的位置
#         self._parse_list_contrastive()  # 此处是处理加入对比数据数据信息提取
#         print()
#
#     def _load_image(self, directory, idx):
#         if self.modality == 'RGB' or self.modality == 'RGBDiff':        # 也就是说只有RGB和Flow的图像，RGBDiff是通过RGB图像计算得到
#             return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
#         elif self.modality == 'Flow':
#             x_img = Image.open(os.path.join(directory, self.image_tmpl.format('flow_x', idx))).convert('L')
#             y_img = Image.open(os.path.join(directory, self.image_tmpl.format('flow_y', idx))).convert('L')
#
#             return [x_img, y_img]   # need return idx as time mark
#
#     def _parse_list(self):  # 此函数是处理正常的TSN数据
#         self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]
#
#     def _parse_list_contrastive(self):  # 此处是加入对比视频数据时数据的处理方式
#         # with open(self.list_file) as f:
#         #     info = f.readlines()
#         # vid_info = [[] for i in range(len(info))]
#         # for i in range(len(info)):
#         #     vid_info[i].extend(info[i].strip().split(' '))
#         # enhance_label = {2: 67, 16: 17, 19: 77, 22: 28, 23: 22, 28: 84, 33: 12, 37: 36, 44: 84, 60: 65, 70: 16, 80: 79, 98: 90}
#         # vid_pairs = 3
#         # new_vid_info = []
#         # for vid in vid_info:
#         #     second_vid_list = []
#         #     if int(vid[2]) in enhance_label.keys():
#         #         for second_vid in vid_info:
#         #             if int(second_vid[2]) == enhance_label[int(int(vid[2]))]:
#         #                 second_vid_list.append(second_vid)
#         #     else:
#         #         second_vid_list.append(vid_info[np.random.randint(0, len(vid_info)-1)])
#         #     for i in range(vid_pairs):
#         #         new_vid_info.append([vid, second_vid_list[np.random.randint(0, len(second_vid_list))]])
#         new_vid_info = funs.read_txt_info(self.list_file)
#         # self.video_list = [VideoRecord_Contrastive(x) for x in new_vid_info]
#         self.video_list = [[] for i in range(len(new_vid_info))]  # self.video_list用来存储视频对的信息
#         for vid_index in range(len(new_vid_info)):  # 第一层循环是每个视频
#             for vid in new_vid_info[vid_index]:  # 第二层循环是每个视频中包含的每个视频
#                 self.video_list[vid_index].append(VideoRecord(vid))
#         print()
#
#
#     def _sample_indices(self, record):
#         """
#
#         :param record: VideoRecord
#         :return: list
#         """
#
#         average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
#         if average_duration > 0:
#             offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
#         elif record.num_frames > self.num_segments:
#             offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
#         else:
#             offsets = np.zeros((self.num_segments,))
#         return offsets + 1
#
#     def _get_val_indices(self, record):
#         if record.num_frames > self.num_segments + self.new_length - 1:
#             tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
#             offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
#         else:
#             offsets = np.zeros((self.num_segments,))
#         return offsets + 1
#
#     def _get_test_indices(self, record):
#
#         tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
#
#         offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
#
#         return offsets + 1
#
#     def __getitem__(self, index):
#         record = self.video_list[index]
#
#         if not self.test_mode:
#             segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
#         else:
#             segment_indices = self._get_test_indices(record)
#
#         return self.get(record, segment_indices)
#
#     def get(self, record, indices):
#
#         images = list()
#         for seg_ind in indices:
#             p = int(seg_ind)
#             for i in range(self.new_length):
#                 seg_imgs = self._load_image(record.path, p)
#                 images.extend(seg_imgs)
#                 if p < record.num_frames:
#                     p += 1
#
#         process_data = self.transform(images)
#         return process_data, record.label
#
#     def __len__(self):
#         return len(self.video_list)

class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file, args,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.args = args
        self.back_train = ''  # 设置类属性 back_train代表反向flow的路径 设置为空
        if args.back_train:  # 根据参数是否传入赋予值
            self.back_train = args.back_train

        if self.modality == 'RGBDiff':
            self.new_length += 1        # Diff needs one more image to calculate diff

        if self.args.contrastive:
            self._parse_list_contrastive()  # 此处是处理加入对比数据数据信息提取
        else:
            self._parse_list()  # 此处是调用正常处理TSN数据的位置



    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':        # 也就是说只有RGB和Flow的图像，RGBDiff是通过RGB图像计算得到
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('flow_x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('flow_y', idx))).convert('L')

            return [x_img, y_img]   # need return idx as time mark

    def _parse_list(self):  # 此函数是处理正常的TSN数据
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]
        if self.back_train and self.modality == 'Flow':  # 同时满足有反向flow文件和模态为flow的条件
            self.back_video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.back_train)]

    def _parse_list_contrastive(self):  # 此处是加入对比视频数据时数据的处理方式
        if self.test_mode:
            self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]
        else:
            new_vid_info = funs.read_txt_info(self.list_file, self.test_mode)
            # self.video_list = [VideoRecord_Contrastive(x) for x in new_vid_info]
            self.video_list = [[] for i in range(len(new_vid_info))]  # self.video_list用来存储视频对的信息
            for vid_index in range(len(new_vid_info)):  # 第一层循环是每个视频
                for vid in new_vid_info[vid_index]:  # 第二层循环是每个视频中包含的每个视频
                    self.video_list[vid_index].append(VideoRecord(vid))


    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def _contrastive_getitem(self, record):
        if self.test_mode:
            segment_indices = self._get_test_indices(record)
            return self.get(record, segment_indices)
        else:
            img = []
            label = []
            for i in record:  # 针对每组数据中的每个值进行获取数据
                get_val = self.get_segment_indices(i)
                img.append(get_val[0])
                label.append(get_val[1])
            segment_index = [img, label]
            return segment_index

    def _common_getitem(self, record):
        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def __getitem__(self, index):  # dataloader加载数据的时候传入的参数是随机的，对应这里的index
        record = self.video_list[index]
        # if self.args.contrastive:
        #     return self._contrastive_getitem(record)  # 使用对比损失时getitem的写法  这里需要修改返回值不然直接就结束了
        # else:
        #     return self._common_getitem(record)  # 原始方法的getitem
        record_return = self._common_getitem(record)
        if self.back_train and self.modality == 'Flow':  # 在这里加入对反向flow加载的处理
            record_back = self.back_video_list[index]
            record_back_return = self._common_getitem(record_back)
            return [record_return, record_back_return]  # 满足if语句则同时返回两个图像的返回值
        return record_return  # 正常情况只有一个返回值


    # def __getitem__(self, index):
    #     record = self.video_list[index]
    #     if self.test_mode:
    #         segment_indices = self._get_test_indices(record)
    #         return self.get(record, segment_indices)
    #     else:
    #         img = []
    #         label = []
    #         for i in record:  # 针对每组数据中的每个值进行获取数据
    #             get_val = self.get_segment_indices(i)
    #             img.append(get_val[0])
    #             label.append(get_val[1])
    #         segment_index = [img, label]
    #         # segment_index = [self.get_segment_indices(i) for i in record]
    #         return segment_index

    def get_segment_indices(self, record):
        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
