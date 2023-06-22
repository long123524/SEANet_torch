import torch
from tqdm import tqdm
import numpy as np
import torchvision
import time
import argparse
from scipy import stats
from losses import BCEDiceLoss
from skimage import morphology

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def align_dims(np_input, expected_dims=2):
    dim_input = len(np_input.shape)
    np_output = np_input
    if dim_input>expected_dims:
        np_output = np_input.squeeze(0)
    elif dim_input<expected_dims:
        np_output = np.expand_dims(np_input, 0)
    assert len(np_output.shape) == expected_dims
    return np_output

def binary_accuracy(pred, label):
    pred = align_dims(pred, 2)
    label = align_dims(label, 2)
    pred = (pred >= 0.5)
    label = (label >= 0.5)

    TP = float((pred * label).sum())
    FP = float((pred * (1 - label)).sum())
    FN = float(((1 - pred) * (label)).sum())
    TN = float(((1 - pred) * (1 - label)).sum())
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    IoU = TP / (TP + FP + FN + 1e-10)
    acc = (TP + TN) / (TP + FP + FN + TN)
    F1 = 0
    if acc > 0.99 and TP == 0:
        precision = 1
        recall = 1
        IoU = 1
    if precision > 0 and recall > 0:
        F1 = stats.hmean([precision, recall])
    return acc, precision, recall, F1, IoU


def evaluate(device, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()

    acc_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()

    with torch.no_grad():

        for iter, data in enumerate(tqdm(data_loader)):

            _, inputs, targets, _,_ = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs.float())

            output_mask = outputs[0].sigmoid().detach().cpu().numpy().squeeze()
            targets1 = targets.detach().cpu().numpy().squeeze()

            res = np.zeros((256, 256))
            # indices = np.argmax(output_mask, axis=0)
            res[output_mask > 0.5] = 255
            res[output_mask <=0.5] = 0
            # res = morphology.remove_small_objects(res.astype(int), 1000)
            acc, precision, recall, F1, IoU = binary_accuracy(res, targets1)


            acc_meter.update(acc)
            precision_meter.update(precision)
            recall_meter.update(recall)
            F1_meter.update(F1)
            IoU_meter.update(IoU)


            crit = BCEDiceLoss()
            loss =  crit(outputs[0], targets)
            losses.append(loss.item())

        print('avg Acc %.2f, Pre %.2f, Recall %.2f, F1 %.2f, IOU %.2f' % (
            acc_meter.avg * 100, precision_meter.avg * 100, recall_meter.avg * 100, F1_meter.avg * 100,
            IoU_meter.avg * 100))

        writer.add_scalar("Dev_Loss", np.mean(losses), epoch)

        writer.add_scalar("Accuracy",acc_meter.avg * 100, epoch)
        writer.add_scalar("F1", F1_meter.avg * 100, epoch)
        writer.add_scalar("IoU", IoU_meter.avg * 100, epoch)


    return acc_meter.avg * 100, time.perf_counter() - start


def visualize(device, epoch, model, data_loader, writer, val_batch_size, train=True):
    def save_image(image, tag, val_batch_size):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(
            image, nrow=int(np.sqrt(val_batch_size)), pad_value=0, padding=25
        )
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            _, inputs, targets, _,_ = data

            inputs = inputs.to(device)

            targets = targets.to(device)
            outputs = model(inputs.float())

            output_mask = outputs[0].detach().cpu().numpy()
            output_mask[ output_mask>0.5]=1
            output_mask[output_mask <=0.5] = 0
            output_final = torch.from_numpy(output_mask)


            if train == "True":                                               #targets得是numpy
                save_image(targets.float(), "Target_train",val_batch_size)    #iou =  cal_iou(targets, output_final, n_class=2)
                save_image(output_final, "Prediction_train",val_batch_size)   #writer.add_scalar("IOU", iou/4, epoch)
            else:
                save_image(targets.float(), "Target", val_batch_size)
                save_image(output_final, "Prediction", val_batch_size)

            break


def create_train_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument("--train_path", type=str, help="path to img jpg files")
    parser.add_argument("--val_path", type=str, help="path to img jpg files")
    parser.add_argument(
        "--model_type",
        type=str,
        help="select model type: unet,dcan,dmtn,psinet,convmcd",
    )
    parser.add_argument("--object_type", type=str, help="Dataset.")
    parser.add_argument(
        "--distance_type",
        type=str,
        default="dist_mask",             #原先是dist_mask
        help="select distance transform type - dist_mask,dist_contour,dist_signed",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="train batch size")
    parser.add_argument(
        "--val_batch_size", type=int, default=8, help="validation batch size"
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")
    parser.add_argument(
        "--use_pretrained", type=bool, default=False, help="Load pretrained checkpoint."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,                           #之前是None
        help="If use_pretrained is true, provide checkpoint.",
    )
    parser.add_argument("--save_path", type=str, help="Model save path.")

    parser.add_argument('--vgg16_caffe', default='./5stage-vgg.py36pickle', help='Resume VGG-16 Caffe parameters.')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=2e-4, type=float,
                        metavar='W', help='default weight decay')
    parser.add_argument('--stepsize', default=15, type=int,
                        metavar='SS', help='learning rate step size')
    parser.add_argument('--gamma', '--gm', default=0.1, type=float,
                        help='learning rate decay parameter: Gamma')
    parser.add_argument('--lr', '--learning_rate', default=1e-8, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr1', '--learning_rate1', default=1e-2, type=float,
                        metavar='LR1', help='initial learning rate')
    parser.add_argument('--itersize', default=2, type=int,
                        metavar='IS', help='iter size')
    # =============== misc

    return parser


def create_validation_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument(
        "--model_type",
        type=str,
        help="select model type: unet,dcan,dmtn,psinet,convmcd",
    )
    parser.add_argument("--val_path", type=str, help="path to img jpg files")
    parser.add_argument("--model_file", type=str, help="model_file")
    parser.add_argument("--save_path", type=str, help="results save path.")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")

    return parser


###计算IOU
def cal_iou(target, pred, n_class=2):
    """
    target是真实标签，shape为(h,w)，像素值为0，1，2...
    pred是预测结果，shape为(h,w)，像素值为0，1，2...
    n_class:为预测类别数量
    """


    h, w = target.shape
    # 转为one-hot，shape变为(h,w,n_class)
    target_one_hot = np.eye(n_class)[target]
    pred_one_hot = np.eye(n_class)[pred]

    target_one_hot[target_one_hot != 0] = 1
    pred_one_hot[pred_one_hot != 0] = 1
    join_result = target_one_hot * pred_one_hot

    join_sum = np.sum(np.where(join_result == 1))  # 计算相交的像素数量
    pred_sum = np.sum(np.where(pred_one_hot == 1))  # 计算预测结果非0得像素数
    target_sum = np.sum(np.where(target_one_hot == 1))  # 计算真实标签的非0得像素数

    iou = join_sum / (pred_sum + target_sum - join_sum + 1e-6)
    print('iou',iou)

    return iou