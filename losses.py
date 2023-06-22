import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


#自动赋值权重
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=3):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

def rcf_loss(inputs, label):

    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask > 0.5).float()).float() # ==1.
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0.
   # inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs.float(), label.float())

    return 1.*torch.sum(cost)


class RCFloss:
    def __init__(self):

        self.device = ''

    def __call__(self, outputs, targets):

        criterion = sum([rcf_loss(outpu, targets) for outpu in outputs])     #sum([rcf_loss(outpu, targets) for outpu in outputs])

        return criterion


awl = AutomaticWeightedLoss(2)



#标签平滑
class LabelSmoothing(nn.Module):
    '''
    Description: 借鉴标签平滑的思想，针对样本中的预假设的 `hard sample`（图像边缘、不同类别交界） 进行标签平滑；
                 平滑因子可指定 smoothing 固定，或在训练过程中，在图像边缘、类间交界设置一定大小过渡带，统计过渡带
                 内像素 `hard sample` 比例动态调整。
    Args (type):
        win_size (int): 过渡带窗口大小；
        num_classes (int): 总类别数目，本次实验类别数为5；
        smoothing (float): 默认值为0.1，若指定 fix_smoothing ，则固定训练过程固定平滑因子为 smoothing。
    '''

    def __init__(self, win_size=5, num_classes=2, smoothing=0.1, fix_smoothing=False):
        super(LabelSmoothing, self).__init__()
        self.fix_smoothing = fix_smoothing
        assert (win_size % 2) == 1
        self.smoothing = smoothing / (num_classes - 1)
        self.win_size = win_size
        self.num_classes = num_classes

        self.find_edge_Conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=win_size,
                                        padding=(win_size - 1) // 2, stride=1, bias=False)
        self.find_edge_Conv.weight.requires_grad = False
        new_state_dict = OrderedDict()
        weight = torch.zeros(1, 1, win_size, win_size)
        weight = weight - 1
        weight[:, :, win_size // 2, win_size // 2] = win_size * win_size - 1
        new_state_dict['weight'] = weight
        self.find_edge_Conv.load_state_dict(new_state_dict)

    def to_categorical(self, y, alpha=0.05, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical = categorical + alpha
        categorical[np.arange(n), y] = (1 - alpha) + (alpha / self.num_classes)
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        categorical = categorical.transpose(0, 3, 1, 2)
        return categorical

    def forward(self, x, target):
        assert x.size(1) == self.num_classes
        log_p = nn.functional.log_softmax(x, dim=1)

        if not self.fix_smoothing:
            self.find_edge_Conv.cuda(device=target.device)
            edge_mask = self.find_edge_Conv(target)
            edge_mask = edge_mask.data.cpu().numpy()
            edge_mask[edge_mask != 0] = 1
            self.smoothing = np.mean(edge_mask)
            if self.smoothing > 0.2:
                self.smoothing = 0.2

        target = target.squeeze(dim=1)
        target = target.data.cpu().numpy()
        onehot_mask = self.to_categorical(target, 0, num_classes=self.num_classes)
        onehot_mask = onehot_mask * (1 - edge_mask)
        softlabel_mask = self.to_categorical(target, alpha=self.smoothing, num_classes=self.num_classes)
        softlabel_mask = softlabel_mask * edge_mask
        onehot_mask = torch.from_numpy(onehot_mask).cuda(device=log_p.device).float()
        softlabel_mask = torch.from_numpy(softlabel_mask).cuda(device=log_p.device).float()
        loss = torch.sum(onehot_mask * log_p + softlabel_mask * log_p, dim=1).mean()
        return -loss



