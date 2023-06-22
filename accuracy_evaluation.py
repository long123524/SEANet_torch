import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torch.utils.data import DataLoader
from dataset import DatasetImageMaskContourDist
import glob
from models import SEANet
from tqdm import tqdm
import numpy as np
import cv2
from utils import create_validation_arg_parser
from scipy import stats
from skimage import morphology

def align_dims(np_input, expected_dims=2):
    dim_input = len(np_input.shape)
    np_output = np_input
    if dim_input > expected_dims:
        np_output = np_input.squeeze(0)
    elif dim_input < expected_dims:
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


def build_model(model_type):
    if model_type == "rcf":
        model = SEANet(num_classes=2)
    return model


if __name__ == "__main__":

    args = create_validation_arg_parser().parse_args()
    args.distance_type = 'dist_contour'
    args.val_path = './Denmark_yz/image'
    val_path = os.path.join(args.val_path, "*.tif")
    model_file = args.model_file = './DM_5706/model/100.pt'
    save_path = args.save_path = './Denmark_yz/1'
    model_type = args.model_type = 'rcf'

    f = open('./Denmark_yz/accuracy.txt', 'w+')

    cuda_no = args.cuda_no
    CUDA_SELECT = "cuda:{}".format(cuda_no)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    val_file_names = glob.glob(val_path)
    valLoader = DataLoader(DatasetImageMaskContourDist(val_file_names, args.distance_type))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = build_model(model_type)
    model = model.to(device)
    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
    model.eval()

    acc_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()

    total_iter = len(val_file_names)

    for i, (img_file_name, inputs, targets1, targets2, targets3) in enumerate(
            tqdm(valLoader)
    ):
        inputs = inputs.to(device)
        targets1 = targets1.to(device)

        outputs1, outputs2, outputs3 = model(inputs)
        outputs4,outputs5,outputs6 = model(torch.flip(inputs, [-1]))
        predict_2 = torch.flip(outputs4, [-1])
        outputs7,outputs8,outputs9 = model(torch.flip(inputs, [-2]))
        predict_3 = torch.flip(outputs7, [-2])
        outputs10,outputs11,outputs12  = model(torch.flip(inputs, [-1, -2]))
        predict_4 = torch.flip(outputs10, [-1, -2])

        predict_list = outputs1 + predict_2 + predict_3 + predict_4

        pred = predict_list/4

        outputs1 = pred.detach().cpu().numpy().squeeze()
        targets1 = targets1.detach().cpu().numpy().squeeze()

        res = np.zeros((256, 256))
        res[outputs1>0.5] = 255
        res[outputs1<=0.5] = 0
        res = morphology.remove_small_objects(res.astype(int), 1000)
        acc, precision, recall, F1, IoU = binary_accuracy(res, targets1)

        acc_meter.update(acc)
        precision_meter.update(precision)
        recall_meter.update(recall)
        F1_meter.update(F1)
        IoU_meter.update(IoU)

        res = np.array(res, dtype='uint8')

        output_path = os.path.join(
            save_path,  os.path.basename(img_file_name[0])
        )
        cv2.imwrite(output_path, res)

        print('Eval num %d/%d, Acc %.2f, precision %.2f, recall %.2f, F1 %.2f, IoU %.2f' % (
            i, total_iter, acc * 100, precision * 100, recall * 100, F1 * 100, IoU * 100))

        f.write('Eval num %d/%d, Acc %.2f, precision %.2f, recall %.2f, F1 %.2f, IoU %.2f\n' % (
            i, total_iter, acc * 100, precision * 100, recall * 100, F1 * 100, IoU * 100))

    print('avg Acc %.2f, Pre %.2f, Recall %.2f, F1 %.2f, IOU %.2f' % (
        acc_meter.avg * 100, precision_meter.avg * 100, recall_meter.avg * 100, F1_meter.avg * 100,
        IoU_meter.avg * 100))
