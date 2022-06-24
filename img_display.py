# -*- coding: utf-8 -*-
# @Time : 2022/6/13 20:17
# @Author : 冷川江
# @Site : 
# @File : img_display.py
# import matplotlib
# import matplotlib.image as mpimg
# from PIL import Image
# import matplotlib.pyplot as plt
# import transforms
# # matplotlib.use('Agg')
# input_size = 224
# img_mul = []
# # img = mpimg.imread('/data/ange/UCF101/images/v_BoxingPunchingBag_g02_c01/img_00007.jpg')
# img_path = '/data/ange/UCF101/images/v_BoxingPunchingBag_g02_c01/img_00007.jpg'
# img = Image.open(img_path).convert('RGB')
# # img_mul.extend(img)
# # img_mul.extend(img)
# trans_func = transforms.GroupMultiScaleCrop(input_size, [1, .875, .75, .66])
# print(img.size)
# plt.subplot(2,1,1)
# plt.imshow(img)
# plt.axis('off') # 不显示坐标轴
# img = trans_func(img)
# # img_new = trans_func(img_mul)
# plt.subplot(2,1,2)
# plt.imshow(img)
#
# plt.show()

import matplotlib
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import transforms
# matplotlib.use('Agg')
input_size = 224
img_mul = []
# img = mpimg.imread('/data/ange/UCF101/images/v_BoxingPunchingBag_g02_c01/img_00007.jpg')
img_path = '/data/ange/UCF101/images/v_BoxingPunchingBag_g02_c01/img_00007.jpg'
img = Image.open(img_path).convert('RGB')
# img.show()

trans_func = transforms.GroupMultiScaleCrop(input_size, [1, .875, .75, .66])
print(img.size)
plt.subplot(2,1,1)
# img.show()
plt.figure('box')
plt.subplot(2,1,1)
plt.imshow(img)
plt.axis('off') # 不显示坐标轴
img = trans_func([img])
# img_new = trans_func(img_mul)
plt.subplot(2,1,2)
plt.imshow(img[0])
print('changed img size:', img[0].size)
# img.show()
# img[0].show()
plt.show()
