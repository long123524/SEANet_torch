from PIL import Image
import os
import glob
from PIL import ImageEnhance
import sys

img_path = r'D:\LJ2\SBA2\SD_train\926_new_train\2'           #输入和输出影像所在文件夹

def get_image_paths(folder):
    return glob.glob(os.path.join(folder, '*.tif'))


def create_read_img(filename):
    # 读取图像
    im = Image.open(filename)

    out_h = im.transpose(Image.FLIP_LEFT_RIGHT)     #水平翻转
    out_w = im.transpose(Image.FLIP_TOP_BOTTOM)    #垂直翻转
    out_90 = im.transpose(Image.ROTATE_90)              #顺时针选择90度
    # out_180 = im.transpose(Image.ROTATE_180)
    # out_270 = im.transpose(Image.ROTATE_270)

    # 亮度增强
    enh_bri = ImageEnhance.Brightness(im)
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    image_brightened.save(filename[:-4] + '_brighter.tif')
    #
    # # 色度增强
    # enh_col = ImageEnhance.Color(im)
    # color = 1.5
    # image_colored = enh_col.enhance(color)
    # image_colored.save(filename[:-4] + '_color.tif')
    #
    # # 对比度增强
    enh_con = ImageEnhance.Contrast(im)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    image_contrasted.save(filename[:-4] + '_contrast.tif')

    # 锐度增强
    # enh_sha = ImageEnhance.Sharpness(im)
    # sharpness = 3.0
    # image_sharped = enh_sha.enhance(sharpness)
    # image_sharped.save(filename[:-4] + '_sharp.tif')

    #
    out_h.save(filename[:-4] + '_h.tif')
    out_w.save(filename[:-4] + '_w.tif')
    out_90.save(filename[:-4] + '_90.tif')
    # out_180.save(filename[:-4] + '_180.tif')
    # out_270.save(filename[:-4] + '_270.tif')

    #print(filename)
imgs = get_image_paths(img_path)
for i in imgs:
    create_read_img(i)
