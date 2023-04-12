# from PIL import Image
# import os
# import glob
# from PIL import ImageEnhance
# import sys
#
# img_path = r'G:\BsiNet\train\mask'           #输入和输出影像所在文件夹
#
# def get_image_paths(folder):
#     return glob.glob(os.path.join(folder, '*.tif'))
#
#
# def create_read_img(filename):
#     # 读取图像
#     print(filename)
#     im = Image.open(filename)
#
#     out_h = im.transpose(Image.FLIP_LEFT_RIGHT)     #水平翻转
#     out_w = im.transpose(Image.FLIP_TOP_BOTTOM)    #垂直翻转
#     out_90 = im.transpose(Image.ROTATE_90)              #顺时针选择90度
#     # out_180 = im.transpose(Image.ROTATE_180)
#     # out_270 = im.transpose(Image.ROTATE_270)
#
#     # 亮度增强
#     enh_bri = ImageEnhance.Brightness(im)
#     brightness = 1.5
#     image_brightened = enh_bri.enhance(brightness)
#     image_brightened.save(filename[:-4] + '_brighter.tif')
#     #
#     # # 色度增强
#     enh_col = ImageEnhance.Color(im)
#     color = 1.5
#     image_colored = enh_col.enhance(color)
#     image_colored.save(filename[:-4] + '_color.tif')
#     #
#     # # 对比度增强
#     # enh_con = ImageEnhance.Contrast(im)
#     # contrast = 1.5
#     # image_contrasted = enh_con.enhance(contrast)
#     # image_contrasted.save(filename[:-4] + '_contrast.tif')
#
#     # 锐度增强
#     # enh_sha = ImageEnhance.Sharpness(im)
#     # sharpness = 3.0
#     # image_sharped = enh_sha.enhance(sharpness)
#     # image_sharped.save(filename[:-4] + '_sharp.tif')
#
#     #
#     out_h.save(filename[:-4] + '_h.tif')
#     out_w.save(filename[:-4] + '_w.tif')
#     out_90.save(filename[:-4] + '_90.tif')
#     # out_180.save(filename[:-4] + '_180.tif')
#     # out_270.save(filename[:-4] + '_270.tif')
#
#     #print(filename)
# imgs = get_image_paths(img_path)
# for i in imgs:
#     create_read_img(i)
#
#
import numpy
from numpy import *
import numpy as np

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# import pylab as plt

fdata = "G:/BsiNet/tobacco_ninghua.txt"



class chj_data(object):
    def __init__(self, data, target):
        self.data = data
        self.target = target


def chj_load_file(fdata):
    data = numpy.loadtxt(fdata, dtype=float32)
    target = data[:,-1:]
    res = chj_data(data, target)
    return res


iris = chj_load_file(fdata)
# exit()
X_tsne = ts = TSNE(n_components=2, init='pca', random_state=100).fit_transform(iris.data)
# X_pca = PCA().fit_transform(iris.data)
print("finish!")
plt.figure(figsize=(10, 6))

labels = iris.target
colors = ['b', 'c']
idx_1 = [i1 for i1 in range(len(labels)) if labels[i1] == 0]
flg1 = plt.scatter(X_tsne[idx_1, 0], X_tsne[idx_1, 1], 20, color=colors[0], label='non-tabacco');
idx_2 = [i2 for i2 in range(len(labels)) if labels[i2] == 1]
flg2 = plt.scatter(X_tsne[idx_2, 0], X_tsne[idx_2, 1], 20, color=colors[1], label='tabacco');

# plt.subplot(121)
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target)
plt.legend()
# plt.xticks([])  # 去掉横坐标值
# plt.yticks([])  # 去掉纵坐标值
plt.savefig('tabacco.png',dpi=300)
plt.show()
# plt.colorbar()  展示渐变图斑的图例