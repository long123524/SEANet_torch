from torch import nn
import torch
from torch.nn import functional as F
import numpy as np


######ASPP模块

class ASPPConv1x1(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        """
        ASPP用的5个处理之1，1个1x1卷积
        :param in_channels: 输入channels，是backbone产生的主要特征的输出channels
        :param out_channels: 输出channels，论文建议取值256
        """
        modules = [nn.Conv2d(in_channels, out_channels, 1, bias=False),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(inplace=True), ]
        super(ASPPConv1x1, self).__init__(*modules)
        pass

    pass


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        """
        ASPP用的5个处理之3，3个dilation conv，都是3x3的same卷积
        :param in_channels: dilation conv的输入channels，是backbone产生的主要特征的输出channels
        :param out_channels: dilation conv的输出channels，论文建议取值256
        :param dilation: 膨胀率，论文建议取值6,12,18
        """
        modules = [nn.Conv2d(in_channels, out_channels, kernel_size=3,
                             padding=dilation, dilation=dilation, bias=False),  # same卷积padding=dilation*(k-1)/2
                   nn.BatchNorm2d(out_channels),  # 有BN，卷积bias=False
                   nn.ReLU(inplace=True), ]  # 激活函数
        super(ASPPConv, self).__init__(*modules)
        pass

    pass


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        """
        ASPP用的5个处理之1，Image Pooling
        :param in_channels: 输入channels，是backbone产生的主要特征的输出channels
        :param out_channels: 输出channels，论文建议取值256
        """
        modules = [nn.AdaptiveAvgPool2d(1),  # 全局平均池化，输出spatial大小1
                   nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # 1x1卷积调整channels
                   nn.BatchNorm2d(out_channels),  # 有BN，卷积bias=False
                   nn.ReLU(inplace=True), ]  # 激活函数
        super(ASPPPooling, self).__init__(*modules)
        pass

    def forward(self, x):
        size = x.shape[-2:]  # 记录下输入的大小
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)  # 双线性差值上采样到原spatial大小

    pass


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        ASPP，对backbone产生的主干特征进行空间金字塔池化。
        金字塔有5层：1个1x1卷积，3个3x3 dilation conv，1个全局平均池化
        将5层cat后再调整channels输出。
        这里不进行upsample，因为不知道low-level的spatial大小。
        :param in_channels: 输入channels，是backbone产生的主要特征的输出channels
        :param out_channels: 输出channels，论文建议取值256
        """
        super(ASPP, self).__init__()
        modules = [ASPPConv1x1(in_channels, out_channels),  # 1个1x1卷积
                   ASPPConv(in_channels, out_channels, dilation=1),  # 3x3 dilation conv，dilation=6
                   ASPPConv(in_channels, out_channels, dilation=2),  # 3x3 dilation conv，dilation=12
                   ASPPConv(in_channels, out_channels, dilation=4),  # 3x3 dilation conv，dilation=18
                   ASPPPooling(in_channels, out_channels), ]  # 全局平均池化Image Pooling
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout2d(0.5))  # 将5层cat后再调整channels输出，但不知道为什么Dropout
        pass

    def forward(self, x):
        output = []
        for mod in self.convs:
            output.append(mod(x))
            pass
        x = torch.cat(output, dim=1)
        x = self.project(x)
        return x

    pass

#######################################

#CBAM注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)





#SE注意力机制
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels,middle_channels,out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, out_channels),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
              #  SELayer(out_channels)          #加的
            )

    def forward(self, x):
        return self.block(x)






# Based on BDCN Implementation @ https://github.com/pkuCactus/BDCN
def crop(data1, h, w, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    assert (h <= h1 and w <= w1)
    data = data1[:, :, crop_h:crop_h + h, crop_w:crop_w + w]
    return data


def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w


def upsample(input, stride, num_channels=1):
    kernel_size = stride * 2
    kernel = make_bilinear_weights(kernel_size, num_channels).cuda()
    return torch.nn.functional.conv_transpose2d(input, kernel, stride=stride)




########
class SEANet(nn.Module):
    def __init__(self,num_classes=2, add_output=True, bilinear=False, num_filters=32,is_deconv=False):
        super(SEANet, self).__init__()
        # lr 1 2 decay 1 0
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)

        self.activ = nn.ReLU(inplace=True)

        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)
        self.maxpool_1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool_2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool_3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        #
        self.center = DecoderBlockV2(512, num_filters * 8*2, num_filters * 8,is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 , num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 4, num_filters * 4*2, num_filters * 4, is_deconv)

        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool1 = nn.MaxPool2d(4, stride=4, ceil_mode=True)
     #   self.xdf = ConvRelu(192,64)
        self.aspp = ASPP(512,256)
        self.aspp1 = ASPP(192,64)


        # lr 0.1 0.2 decay 1 0
        self.conv1_1_down = nn.Conv2d(64, 21, 1, padding=0)
        self.conv1_2_down = nn.Conv2d(64, 21, 1, padding=0)

        self.conv2_1_down = nn.Conv2d(128, 21, 1, padding=0)
        self.conv2_2_down = nn.Conv2d(128, 21, 1, padding=0)

        self.conv3_1_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_2_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_3_down = nn.Conv2d(256, 21, 1, padding=0)

        self.conv4_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_3_down = nn.Conv2d(512, 21, 1, padding=0)

        self.conv5_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_3_down = nn.Conv2d(512, 21, 1, padding=0)

        # lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        # lr 0.001 0.002 decay 1 0
        self.score_final = nn.Conv2d(5, 1, 1)

        ## Fixed the upsampling weights for the training process as per @https://github.com/xwjabc/hed
        self.weight_deconv2 = make_bilinear_weights(4, 1).cuda()
        self.weight_deconv3 = make_bilinear_weights(8, 1).cuda()
        self.weight_deconv4 = make_bilinear_weights(16, 1).cuda()
        # Wrong Deconv Filter size. Updated from RCF yun_liu
        self.weight_deconv5 = make_bilinear_weights(16, 1).cuda()
        self.add_output = add_output
        if add_output:
            self.conv_final1 = nn.Conv2d(64, 1, 1)
            self.conv_final2 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]

        conv1_1 = self.activ(self.conv1_1(x))
        conv1_2 = self.activ(self.conv1_2(conv1_1))
        pool1 = self.maxpool_1(conv1_2)

        conv2_1 = self.activ(self.conv2_1(pool1))
        conv2_2 = self.activ(self.conv2_2(conv2_1))
        pool2 = self.maxpool_2(conv2_2)

        conv3_1 = self.activ(self.conv3_1(pool2))
        conv3_2 = self.activ(self.conv3_2(conv3_1))
        conv3_3 = self.activ(self.conv3_3(conv3_2))
        pool3 = self.maxpool_3(conv3_3)

        conv4_1 = self.activ(self.conv4_1(pool3))
        conv4_2 = self.activ(self.conv4_2(conv4_1))
        conv4_3 = self.activ(self.conv4_3(conv4_2))
        pool4 = self.maxpool4(conv4_3)

     #   pool10 = self.maxpool(conv4_3)

        conv5_1 = self.activ(self.conv5_1(pool4))
        conv5_2 = self.activ(self.conv5_2(conv5_1))
        conv5_3 = self.activ(self.conv5_3(conv5_2))

        conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.conv1_2_down(conv1_2)
        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.conv4_3_down(conv4_3)
        conv5_1_down = self.conv5_1_down(conv5_1)
        conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.conv5_3_down(conv5_3)

        center = self.aspp(conv4_3)

        dec5 = self.dec5(torch.cat([center, conv4_3], 1))

        dec4 = self.dec4(torch.cat([dec5, conv3_3], 1))
        dec3 = self.dec3(torch.cat([dec4, conv2_2], 1))
        dec1 = self.aspp1(torch.cat([dec3, conv1_2], 1))


        so1_out = self.score_dsn1(conv1_1_down + conv1_2_down)
        so2_out = self.score_dsn2(conv2_1_down + conv2_2_down)
        so3_out = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
        so4_out = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
        so5_out = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down)

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, self.weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, self.weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, self.weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, self.weight_deconv5, stride=8)

        ### center crop
        so1 = crop(so1_out, img_H, img_W, 0, 0)
        so2 = crop(upsample2, img_H, img_W, 1, 1)
        so3 = crop(upsample3, img_H, img_W, 2, 2)
        so4 = crop(upsample4, img_H, img_W, 4, 4)
        so5 = crop(upsample5, img_H, img_W, 0, 0)

        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fuse = self.score_final(fusecat)

        if self.add_output:
            x_out1 = self.conv_final1(dec1)
            x_out3 = self.conv_final2(dec1)

            results = [so1, so2, so3, so4, so5, fuse]
            x_out2 = [torch.sigmoid(r) for r in results]
            x_out3 = torch.sigmoid(x_out3)

        return [x_out1,x_out2,x_out3]

