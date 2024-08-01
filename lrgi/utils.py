import os
import os.path as osp
import time
import random
import numpy as np
import torch

from .scheduler import CosineDecayScheduler, ConstantScheduler

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
def get_optimizer(parameters, lr, wd=0.0):
    if wd > 0.0:
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=wd)
    else:
        return torch.optim.Adam(parameters, lr=lr)

def get_scheduler(lr, lr_warmup_epochs, epochs):
    return CosineDecayScheduler(lr, lr_warmup_epochs, epochs) if lr_warmup_epochs is not None else ConstantScheduler(lr)

def update_lr(step, scheduler, optimizer):
    lr = scheduler.get(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_ckpt(folder, ckpt, name=None):
    if not osp.exists(folder):
        os.makedirs(folder)
    if name is None:
        name = "ckpt_{}".format( len(os.listdir(folder)) )
    path = osp.join(folder, name + "pt")
    torch.save(ckpt, path)
    return path    