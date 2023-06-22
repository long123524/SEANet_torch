import torch
import numpy as np
import cv2
from PIL import Image, ImageFile
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import io
from torch.optim.optimizer import Optimizer
import os



class DatasetImageMaskContourDist(Dataset):

    def __init__(self, dir, file_names, distance_type):

        self.file_names = file_names
        self.distance_type = distance_type
        self.dir = dir


    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, idx):

        img_file_name = self.file_names[idx]
        image = load_image(os.path.join(self.dir,img_file_name+'.tif'))
        #print(image.shape)
        mask = load_mask(os.path.join(self.dir,img_file_name+'.tif'))
        #print(img_file_name)
        contour = load_contour(os.path.join(self.dir,img_file_name+'.tif'))
        dist = load_distance(os.path.join(self.dir,img_file_name+'.tif'), self.distance_type)

        return img_file_name, image, mask, contour,  dist


def load_image(path):

    img = Image.open(path)
    data_transforms = transforms.Compose(
        [
         #   transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ]
    )
    img = data_transforms(img)
    return img


def load_mask(path):

    mask = cv2.imread(path.replace("image", "mask").replace("tif", "tif"), 0)
    mask = mask/255.

    return torch.from_numpy(np.expand_dims(mask, 0)).float()


def load_contour(path):

    lb = cv2.imread(path.replace("image", "contour").replace("tif", "tif"), 0)
    lb = lb/255.

    return torch.from_numpy(np.expand_dims(lb, 0)).float()


def load_distance(path, distance_type):

    if distance_type == "dist_mask":
        path = path.replace("image", "dist_mask").replace("tif", "mat")
        #print(path)
        #dist = io.loadmat(path)["mask_dist"]
        dist = io.loadmat(path)["D2"]


    if distance_type == "dist_contour":
        dist = cv2.imread(path.replace("image", "dist_contour").replace("tif", "tif"), 0)
        dist = dist/255
       # dist = io.loadmat(path)["D2"]
    if distance_type == "dist_signed":
        path = path.replace("image", "dist_signed").replace("jpg", "mat")
        dist = io.loadmat(path)["dist_norm"]

    return torch.from_numpy(np.expand_dims(dist, 0)).float()



class Apollo(Optimizer):
    r"""Implements Atom algorithm.
        Arguments:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float): learning rate
            beta (float, optional): coefficient used for computing running averages of gradient (default: 0.9)
            eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-4)
            rebound (str, optional): recified bound for diagonal hessian:
                ``'constant'`` | ``'belief'`` (default: None)
            warmup (int, optional): number of warmup steps (default: 500)
            init_lr (float, optional): initial learning rate for warmup (default: lr/1000)
            weight_decay (float, optional): weight decay coefficient (default: 0)
            weight_decay_type (str, optional): type of weight decay:
                ``'L2'`` | ``'decoupled'`` | ``'stable'`` (default: 'L2')
        """

    def __init__(self, params, lr, beta=0.9, eps=1e-4, rebound='constant', warmup=500, init_lr=1e-5, weight_decay=1e-4, weight_decay_type=None):
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate value: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta))
        if rebound not in ['constant', 'belief']:
            raise ValueError("Invalid recitifed bound: {}".format(rebound))
        if not 0.0 <= warmup:
            raise ValueError("Invalid warmup updates: {}".format(warmup))
        if init_lr is None:
            init_lr = lr / 1000
        if not 0.0 <= init_lr <= lr:
            raise ValueError("Invalid initial learning rate: {}".format(init_lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if weight_decay_type is None:
            weight_decay_type = 'L2' if rebound == 'constant' else 'decoupled'
        if weight_decay_type not in ['L2', 'decoupled', 'stable']:
            raise ValueError("Invalid weight decay type: {}".format(weight_decay_type))

        defaults = dict(lr=lr, beta=beta, eps=eps, rebound=rebound,
                        warmup=warmup, init_lr=init_lr, base_lr=lr,
                        weight_decay=weight_decay, weight_decay_type=weight_decay_type)
        super(Apollo, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Apollo, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['approx_hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Previous update direction
                    state['update'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Calculate current lr
                if state['step'] < group['warmup']:
                    curr_lr = (group['base_lr'] - group['init_lr']) * state['step'] / group['warmup'] + group['init_lr']
                else:
                    curr_lr = group['lr']

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Atom does not support sparse gradients.')

                # Perform step weight decay
                if group['weight_decay'] != 0 and group['weight_decay_type'] == 'L2':
                    grad = grad.add(p, alpha=group['weight_decay'])

                beta = group['beta']
                eps = group['eps']
                exp_avg_grad = state['exp_avg_grad']
                B = state['approx_hessian']
                d_p = state['update']

                state['step'] += 1
                bias_correction = 1 - beta ** state['step']
                alpha = (1 - beta) / bias_correction

                # calc the diff grad
                delta_grad = grad - exp_avg_grad
                if group['rebound'] == 'belief':
                    rebound = delta_grad.norm(p=np.inf)
                else:
                    rebound = 0.01
                    eps = eps / rebound

                # Update the running average grad
                exp_avg_grad.add_(delta_grad, alpha=alpha)

                denom = d_p.norm(p=4).add(eps)
                d_p.div_(denom)
                v_sq = d_p.mul(d_p)
                delta = delta_grad.div_(denom).mul_(d_p).sum().mul(-alpha) - B.mul(v_sq).sum()

                # Update B
                B.addcmul_(v_sq, delta)

                # calc direction of parameter updates
                if group['rebound'] == 'belief':
                    denom = torch.max(B.abs(), rebound).add_(eps / alpha)
                else:
                    denom = B.abs().clamp_(min=rebound)

                d_p.copy_(exp_avg_grad.div(denom))

                # Perform step weight decay
                if group['weight_decay'] != 0 and group['weight_decay_type'] != 'L2':
                    if group['weight_decay_type'] == 'stable':
                        weight_decay = group['weight_decay'] / denom.mean().item()
                    else:
                        weight_decay = group['weight_decay']
                    d_p.add_(p, alpha=weight_decay)

                p.add_(d_p, alpha=-curr_lr)

        return loss