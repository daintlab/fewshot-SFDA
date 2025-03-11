import numpy as np
import torch
import random
from datetime import datetime
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    
def freeze_except_bn(model,layer=None):
    model.eval()
    count = 0
    if layer is None:
        layer = 53
    if type(layer) == int :
        for m in model.modules():
            if isinstance(m,nn.BatchNorm2d):
                m.train()
                count += 1
                if layer == count:
                    break
    elif type(layer) == list:
        for m in model.modules():
            if isinstance(m,nn.BatchNorm2d):
                if count in layer:
                    m.train()
                    # print(f"{count}-th BN layer activated")
                count += 1
    
    # print(f"Model freezed except {count} BN layers")

def get_bn_params(model,affine,layer=None):
    bn_params = []
    count = 0
    if layer is None:
        layer = 53
    if type(layer) == int:
        for m in model.modules():
            if isinstance(m,nn.BatchNorm2d):
                if affine == 'affine' or affine == 'BN':
                    bn_params.extend(list(m.parameters()))
                elif affine == 'affine-scale':
                    bn_params.append(m.weight)
                elif affine == 'affine-shift':
                    bn_params.append(m.bias)
                count += 1
                if layer == count:
                    break
    elif type(layer) == list:
        for m in model.modules():
            if isinstance(m,nn.BatchNorm2d):
                if count in layer:
                    if affine == 'affine' or affine == 'BN':
                        bn_params.extend(list(m.parameters()))
                    elif affine == 'affine-scale':
                        bn_params.append(m.weight)
                    elif affine == 'affine-shift':
                        bn_params.append(m.bias)
                    # print(f"{count}-th BN layer param collected")
                count += 1
 
    return bn_params

def get_proper_param(model,args):
    if args.adapt == 'cls':
        param = model.netC.parameters()
    elif args.adapt == 'BN':
        param = get_bn_params(model,affine='BN')
    elif args.adapt == 'clsBN':
        param = get_bn_params(model,affine='BN')
        param.extend(list(model.netC.parameters()))
    elif args.adapt == 'feat':
        param = list(model.netF.parameters())+list(model.netB.parameters())
    elif args.adapt == 'all':
        param = model.parameters()
    else:
        raise NotImplementedError
    
    return param
    
def freeze_proper_param(model,args):
    model.eval()
    
    if args.adapt == 'cls':
        model.netC.train()
    elif args.adapt == 'BN':
        freeze_except_bn(model)
    elif args.adapt == 'clsBN':
        freeze_except_bn(model)
        model.netC.train()
    elif args.adapt == 'feat':
        model.netF.train()
        model.netB.train()
    elif args.adapt == 'all':
        model.train()
    else:
        raise NotImplementedError

