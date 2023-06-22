import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import glob
import logging
import random
from dataset import *
from losses import *
from models import SEANet
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import visualize, evaluate, create_train_arg_parser
import pickle
import numpy as np
from collections import defaultdict
from torch.optim import lr_scheduler
import scipy.io as sio
from torchtoolbox.tools import mixup_data, mixup_criterion
from sklearn.model_selection import train_test_split
# from albumentations.augmentations import transforms
# from albumentations.core.composition import Compose, OneOf
# from albumentations import RandomRotate90,Resize,Cutout
# from thop import profile
# from torchinfo import summary
##权重初始化
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # xavier(m.weight.data)
        m.weight.data.normal_(0, 0.01)
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # for new_score_weight
            torch.nn.init.constant_(m.weight, 0.2)
        if m.bias is not None:
            m.bias.data.zero_()


#RCF预训练文件
def load_vgg16pretrain(model, vggmodel='./vgg16convs.mat'):
    vgg16 = sio.loadmat(vggmodel)
    torch_params = model.state_dict()

    for k in vgg16.keys():
        name_par = k.split('-')
        size = len(name_par)
        if size == 2:
            name_space = name_par[0] + '.' + name_par[1]
            data = np.squeeze(vgg16[k])
            torch_params[name_space] = torch.from_numpy(data)
    model.load_state_dict(torch_params)


def build_model(model_type):

    if model_type == "rcf":
        model = SEANet(num_classes=1)
    return model



if __name__ == "__main__":

    args = create_train_arg_parser().parse_args()
    args.distance_type = 'dist_contour'
    args.train_path = r'D:\LJ2\SBA2\DM_5706\image'
    # args.val_path = './XJ_kerl_2/valid/image/'
    args.model_type = 'rcf'
    args.object_type = 'polyp'
    args.save_path = r'D:\LJ2\SBA2\DM_5706\se_dm'
 #   args.pretrained_model_path = './XJ_new_model_xinzeng/15.pt'
    CUDA_SELECT = "cuda:{}".format(args.cuda_no)
    log_path = args.save_path + "/summary"
    writer = SummaryWriter(log_dir=log_path)

    logging.basicConfig(
        filename="".format(args.object_type),
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )
    logging.info("")

    train_file_names = glob.glob(os.path.join(args.train_path, "*.tif"))
    random.shuffle(train_file_names)
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_file_names]
    train_file, val_file = train_test_split(img_ids, test_size=0.2, random_state=41)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")
    print(device)
    model = build_model(args.model_type)
    # print(summary(model,(1,3,256,256)))

    if torch.cuda.device_count() > 0:           #本来是0
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        #model = nn.DataParallel(model)

    model = model.to(device)

    # To handle epoch start number and pretrained weight
    # if args.vgg16_caffe:
    #     load_vgg16_caffe(model, args.vgg16_caffe)        HED
    model.apply(weights_init)
    load_vgg16pretrain(model)  #

   # net_parameters_id = {}
    net_parameters_id = defaultdict(list)
    net = model
    for pname, p in net.named_parameters():
        print(pname)
        if pname in ['conv1_1.weight','conv1_2.weight',
                     'conv2_1.weight','conv2_2.weight',
                     'conv3_1.weight','conv3_2.weight','conv3_3.weight',
                     'conv4_1.weight','conv4_2.weight','conv4_3.weight']:
            # print(pname, 'lr:1 de:1')
            if 'conv1-4.weight' not in net_parameters_id:
                net_parameters_id['conv1-4.weight'] = []
            net_parameters_id['conv1-4.weight'].append(p)

        elif pname in ['conv_final1.weight','conv_final2.weight']:
            # print(pname, 'lr:2 de:0')
            net_parameters_id['final1-2.weight'].append(p)
        elif pname in ['conv_final1.bias','conv_final2.bias']:
            # print(pname, 'lr:2 de:0')
            net_parameters_id['final1-2.bias'].append(p)

        elif pname in ['conv1_1.bias','conv1_2.bias',
                       'conv2_1.bias','conv2_2.bias',
                       'conv3_1.bias','conv3_2.bias','conv3_3.bias',
                       'conv4_1.bias','conv4_2.bias','conv4_3.bias']:
            # print(pname, 'lr:2 de:0')
            if 'conv1-4.bias' not in net_parameters_id:
                net_parameters_id['conv1-4.bias'] = []
            net_parameters_id['conv1-4.bias'].append(p)
        elif pname in ['conv5_1.weight','conv5_2.weight','conv5_3.weight']:
            # print(pname, 'lr:100 de:1')
            if 'conv5.weight' not in net_parameters_id:
                net_parameters_id['conv5.weight'] = []
            net_parameters_id['conv5.weight'].append(p)
        elif pname in ['conv5_1.bias','conv5_2.bias','conv5_3.bias'] :
            # print(pname, 'lr:200 de:0')
            if 'conv5.bias' not in net_parameters_id:
                net_parameters_id['conv5.bias'] = []
            net_parameters_id['conv5.bias'].append(p)
        elif pname in ['conv1_1_down.weight','conv1_2_down.weight',
                       'conv2_1_down.weight','conv2_2_down.weight',
                       'conv3_1_down.weight','conv3_2_down.weight','conv3_3_down.weight',
                       'conv4_1_down.weight','conv4_2_down.weight','conv4_3_down.weight',
                       'conv5_1_down.weight','conv5_2_down.weight','conv5_3_down.weight']:
            # print(pname, 'lr:0.1 de:1')
            if 'conv_down_1-5.weight' not in net_parameters_id:
                net_parameters_id['conv_down_1-5.weight'] = []
            net_parameters_id['conv_down_1-5.weight'].append(p)
        elif pname in ['conv1_1_down.bias','conv1_2_down.bias',
                       'conv2_1_down.bias','conv2_2_down.bias',
                       'conv3_1_down.bias','conv3_2_down.bias','conv3_3_down.bias',
                       'conv4_1_down.bias','conv4_2_down.bias','conv4_3_down.bias',
                       'conv5_1_down.bias','conv5_2_down.bias','conv5_3_down.bias']:
            # print(pname, 'lr:0.2 de:0')
            if 'conv_down_1-5.bias' not in net_parameters_id:
                net_parameters_id['conv_down_1-5.bias'] = []
            net_parameters_id['conv_down_1-5.bias'].append(p)
        elif pname in ['score_dsn1.weight','score_dsn2.weight','score_dsn3.weight',
                       'score_dsn4.weight','score_dsn5.weight']:
            # print(pname, 'lr:0.01 de:1')
            if 'score_dsn_1-5.weight' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.weight'] = []
            net_parameters_id['score_dsn_1-5.weight'].append(p)
        elif pname in ['score_dsn1.bias','score_dsn2.bias','score_dsn3.bias',
                       'score_dsn4.bias','score_dsn5.bias']:
            # print(pname, 'lr:0.02 de:0')
            if 'score_dsn_1-5.bias' not in net_parameters_id:
                net_parameters_id['score_dsn_1-5.bias'] = []
            net_parameters_id['score_dsn_1-5.bias'].append(p)
        elif pname in ['score_final.weight']:
            # print(pname, 'lr:0.001 de:1')
            if 'score_final.weight' not in net_parameters_id:
                net_parameters_id['score_final.weight'] = []
            net_parameters_id['score_final.weight'].append(p)
        elif pname in ['score_final.bias']:
            # print(pname, 'lr:0.002 de:0')
            if 'score_final.bias' not in net_parameters_id:
                net_parameters_id['score_final.bias'] = []
            net_parameters_id['score_final.bias'].append(p)

        elif pname in ['aspp.convs.0.0.weight','aspp.convs.0.1.weight','aspp.convs.1.0.weight','aspp.convs.1.1.weight','aspp.convs.2.0.weight',
                       'aspp.convs.2.1.weight','aspp.convs.3.0.weight','aspp.convs.3.1.weight','aspp.convs.4.1.weight','aspp.convs.4.2.weight',
                       'aspp.project.0.weight','aspp.project.1.weight']:
            # print(pname, 'lr:0.002 de:0')
            if 'aspp1-12.weight' not in net_parameters_id:
                net_parameters_id['aspp1-12.weight'] = []
            net_parameters_id['aspp1-12.weight'].append(p)
        elif pname in ['aspp.convs.0.1.bias','aspp.convs.1.1.bias','aspp.convs.2.1.bias','aspp.convs.3.1.bias',
                       'aspp.convs.4.2.bias','aspp.project.1.bias']:
            # print(pname, 'lr:0.002 de:0')
            if 'aspp1-6.bias' not in net_parameters_id:
                net_parameters_id['aspp1-6.bias'] = []
            net_parameters_id['aspp1-6.bias'].append(p)

        elif pname in ['aspp1.convs.0.0.weight','aspp1.convs.0.1.weight','aspp1.convs.1.0.weight','aspp1.convs.1.1.weight','aspp1.convs.2.0.weight',
                       'aspp1.convs.2.1.weight','aspp1.convs.3.0.weight','aspp1.convs.3.1.weight','aspp1.convs.4.1.weight','aspp1.convs.4.2.weight',
                       'aspp1.project.0.weight','aspp1.project.1.weight']:
            # print(pname, 'lr:0.002 de:0')
            if 'as1-12.weight' not in net_parameters_id:
                net_parameters_id['as1-12.weight'] = []
            net_parameters_id['as1-12.weight'].append(p)
        elif pname in ['aspp1.convs.0.1.bias','aspp1.convs.1.1.bias','aspp1.convs.2.1.bias','aspp1.convs.3.1.bias',
                       'aspp1.convs.4.2.bias','aspp1.project.1.bias']:
            # print(pname, 'lr:0.002 de:0')
            if 'as1-6.bias' not in net_parameters_id:
                net_parameters_id['as1-6.bias'] = []
            net_parameters_id['as1-6.bias'].append(p)

        elif pname in ['center.block.1.conv.weight','dec5.block.1.conv.weight','dec5.block.2.conv.weight','dec4.block.1.conv.weight','dec4.block.2.conv.weight','dec3.block.2.conv.weight','dec3.block.1.conv.weight',
                       'dec2.block.2.conv.weight','dec2.block.1.conv.weight','dec1.conv.weight',' xdf.conv.weight','center.block.2.conv.weight',
                       'center.block.3.fc.0.weight','center.block.3.fc.2.weight','dec5.block.3.fc.0.weight','dec5.block.3.fc.2.weight','dec4.block.3.fc.0.weight',
                       'dec4.block.3.fc.2.weight','dec3.block.3.fc.0.weight','dec3.block.3.fc.2.weight','dec2.block.3.fc.0.weight','dec2.block.3.fc.2.weight']:
            # print(pname, 'lr:0.002 de:0')
            if 'dec1-22.weight' not in net_parameters_id:
                net_parameters_id['dec1-22.weight'] = []
            net_parameters_id['dec1-22.weight'].append(p)

        elif pname in ['center.block.1.conv.bias','dec5.block.1.conv.bias','dec5.block.2.conv.bias','dec4.block.1.conv.bias','dec4.block.2.conv.bias','dec3.block.2.conv.bias','dec3.block.1.conv.bias',
                       'dec2.block.2.conv.bias','dec2.block.1.conv.bias','dec1.conv.bias',' xdf.conv.bias','center.block.2.conv.bias']:
            # print(pname, 'lr:0.002 de:0')
            if 'dec1-12.bias' not in net_parameters_id:
                net_parameters_id['dec1-12.bias'] = []
            net_parameters_id['dec1-12.bias'].append(p)

    optimizer = torch.optim.SGD([
            {'params': net_parameters_id['conv1-4.weight']      , 'lr': args.lr*1    , 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['conv1-4.bias']        , 'lr': args.lr*2    , 'weight_decay': 0.},
            {'params': net_parameters_id['conv5.weight']        , 'lr': args.lr*10  , 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['conv5.bias']          , 'lr': args.lr*20  , 'weight_decay': 0.},
            {'params': net_parameters_id['conv_down_1-5.weight'], 'lr': args.lr*0.1  , 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['conv_down_1-5.bias']  , 'lr': args.lr*0.2  , 'weight_decay': 0.},
            {'params': net_parameters_id['score_dsn_1-5.weight'], 'lr': args.lr*0.01 , 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['score_dsn_1-5.bias']  , 'lr': args.lr*0.02 , 'weight_decay': 0.},
            {'params': net_parameters_id['score_final.weight']  , 'lr': args.lr*0.001, 'weight_decay': args.weight_decay},
            {'params': net_parameters_id['score_final.bias']    , 'lr': args.lr*0.002, 'weight_decay': 0.},
            {'params': awl.parameters(),'lr': args.lr}
        ], lr=args.lr, momentum=args.momentum,  weight_decay=args.weight_decay)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    epoch_start = "0"
    if args.use_pretrained:
        print("Loading Model {}".format(os.path.basename(args.pretrained_model_path)))
        model.load_state_dict(torch.load(args.pretrained_model_path))
        epoch_start = os.path.basename(args.pretrained_model_path).split(".")[0]
        print(epoch_start)
    #print(args.use_pretrained)

    trainLoader = DataLoader(
        DatasetImageMaskContourDist(args.train_path,train_file, args.distance_type),
        batch_size=args.batch_size,drop_last=False, shuffle=True
    )
    devLoader = DataLoader(
        DatasetImageMaskContourDist(args.train_path,val_file, args.distance_type),drop_last=False,shuffle=False
    )
    displayLoader = DataLoader(
        DatasetImageMaskContourDist(args.train_path,val_file, args.distance_type),
        batch_size=args.val_batch_size,drop_last=False,shuffle=False
    )

    optimizer1 = Apollo([{'params': net_parameters_id['final1-2.weight'],'lr': args.lr1},
                                  {'params': net_parameters_id['dec1-22.weight'],'lr': args.lr1},
                                  {'params': net_parameters_id['dec1-12.bias'], 'lr': args.lr1},
                                  {'params': net_parameters_id['final1-2.bias'], 'lr': args.lr1},
                                  {'params': net_parameters_id['aspp1-12.weight'], 'lr': args.lr1},
                                  {'params': net_parameters_id['aspp1-6.bias'], 'lr': args.lr1},
                         {'params': net_parameters_id['as1-12.weight'], 'lr': args.lr1},
                         {'params': net_parameters_id['as1-6.bias'], 'lr': args.lr1}], lr=args.lr1)

  #  scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.stepsize, gamma=args.gamma)
  #  scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, int(1e10), eta_min=1e-5)

    for epoch in tqdm(
        range(int(epoch_start) + 1, int(epoch_start) + 1 + args.num_epochs)
    ):
        global_step = epoch * len(trainLoader)
        running_loss = 0.0

        counter = 0

        for i, (img_file_name, inputs, targets1, targets2,targets3) in enumerate(
            tqdm(trainLoader)
        ):

            model.train()

            inputs = inputs.to(device)
            targets1 = targets1.to(device).float()
            targets2 = targets2.to(device).float()
            targets3 = targets3.to(device).float()
            counter += 1
            ###mix_up，from torchtoolbox.tools import mixup_data, mixup_criterion（pip install torchtoolbox)
            # alpha=0.2
            # data1, labels_a, labels_b, lam1 = mixup_data(inputs, targets1, alpha)
            # data2, labels_c, labels_d, lam2 = mixup_data(inputs, targets2, alpha)
            # data3, labels_e, labels_f, lam3 = mixup_data(inputs, targets3, alpha)
            criterion3 = nn.MSELoss()
            criterion1 = BCEDiceLoss()
            criterion2 = RCFloss()
            with torch.set_grad_enabled(True):
                output1, output2, output3 = model(inputs.float())
                loss1 = criterion1(output1,targets1)
                loss2 = criterion2(output2,targets2)
                loss3 = criterion3(output3,targets3)
                # loss1 = mixup_criterion(criterion1, output1, labels_a, labels_b, lam1)
                # loss2 = mixup_criterion(criterion2, output2, labels_c, labels_d, lam2)
                # loss3 = mixup_criterion(criterion3, output3, labels_e, labels_f, lam3)
                loss = awl(loss1,loss3)
                loss = loss + loss2
                loss = loss / args.itersize
                loss.backward()

            if counter == args.itersize:
                optimizer.step()
                optimizer.zero_grad()
                optimizer1.step()
                optimizer1.zero_grad()
                counter = 0

            writer.add_scalar("loss", loss.item(), epoch)

            running_loss += loss.item() * inputs.size(0)


        epoch_loss = running_loss / len(train_file_names)
        scheduler.step()
      #  scheduler1.step()
      #   print(epoch_loss)

        if epoch % 1 == 0:

            dev_loss, dev_time = evaluate(device, epoch, model, devLoader, writer)
            writer.add_scalar("valid overall accuracy", dev_loss, epoch)
            visualize(device, epoch, model, displayLoader, writer, args.val_batch_size)
            print("Global Loss:{} Val Loss:{}".format(epoch_loss, dev_loss))
        else:
            print("Global Loss:{} ".format(epoch_loss))

        logging.info("epoch:{} train_loss:{} ".format(epoch, epoch_loss))
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(), os.path.join(args.save_path, str(epoch) + ".pt")
            )









