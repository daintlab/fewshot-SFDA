import numpy as np
import torch
import random
from datetime import datetime
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def timestamp(fmt="%y%m%d_%H-%M-%S"):
    return datetime.now().strftime(fmt)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.6f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)
    
def freeze_except_bn(model):
    model.eval()
    count = 0
    for m in model.modules():
        if isinstance(m,nn.BatchNorm2d):
            count += 1
            m.train()
    # print(f"Model freezed except {count} BN layers")

def get_bn_params(model,affine):
    bn_params = []
    for m in model.modules():
        if isinstance(m,nn.BatchNorm2d):
            if affine == 'affine' or affine == 'BN':
                bn_params.extend(list(m.parameters()))
            elif affine == 'affine-scale':
                bn_params.append(m.weight)
            elif affine == 'affine-shift':
                bn_params.append(m.bias)
    
    return bn_params
    