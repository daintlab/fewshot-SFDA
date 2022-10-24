import argparse
import os
import json
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict

from model import ERM
from dataset import get_target_dataset
from data_loader import InfiniteDataLoader
import utils
import warnings

warnings.filterwarnings(action='ignore')


parser = argparse.ArgumentParser(description='Adapt and test on target domain')
parser.add_argument('--data_dir', default='/data2/domainbed',type=str,
                    help='Data directory')
parser.add_argument('--dataset', default='PACS',type=str,
                    help='Data directory')
parser.add_argument('--target', default=None,type=int,
                    help='Index of target domain')
parser.add_argument('--ckpt', default=None,type=str,
                    help='path to source trained model')
parser.add_argument('--save_path', default='./result',type=str,
                    help='Save path')
parser.add_argument('--adapt', default=None,type=str,
                    help='Adapt mode')
parser.add_argument('--batch_size', default=32,type=int,
                    help='Batch size per domain')
parser.add_argument('--lr', default=5e-05,type=float,
                    help='Learning rate')
parser.add_argument('--wd', default=1e-06,type=float,
                    help='Weight decay')
parser.add_argument('--ratio', default=0.2,type=float,
                    help='Holdout ratio')
parser.add_argument('--seed', default=0,type=int,
                    help='Random seed')
parser.add_argument('--adapt_step', default=5000,type=int,
                    help='Adaptation step')
parser.add_argument('--val_freq', default=100,type=int,
                    help='Validation frequency')
parser.add_argument('--severity', default=5,type=int,
                    help='severity of Imagenet-C')
parser.add_argument('--few_shot', default=None,type=int,
                    help='adapt for a few images')
parser.add_argument('--source_type', default='single',type=str,
                    help='type of source domain dataset')
parser.add_argument('--aug', default='default',type=str, choices=['default'],
                    help='Type of augmentation for training stage')
args = parser.parse_args()

def freeze_option(model,adapt):
    if adapt in ['stat','BN']:
        utils.freeze_except_bn(model)
    else:
        model.eval()
    
def adapt(model,criterion,optimizer,data,target):
    freeze_option(model,args.adapt)
    data,target = data.cuda(),target.cuda()
    output = model(data)
    loss = criterion(output,target)
    
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    acc,_= utils.accuracy(output,target,topk=(1,5))
    return loss.item(), acc
        
def test(model,criterion,loader):
    model.eval()
    test_loss = utils.AverageMeter()
    test_acc = utils.AverageMeter()
    with torch.no_grad():
        for i,(x,y) in enumerate(loader):
            x,y = x.cuda(),y.cuda()
            output = model(x)
            loss = criterion(output,y)
            
            acc,_= utils.accuracy(output,y,topk=(1,5))
            test_loss.update(loss.item(),x.size(0))
            test_acc.update(acc[0].item(),x.size(0))
    return test_loss.avg, test_acc.avg

def adapt_to_target(ckpt_path):
    '''
    Adapt source-trained model on train set of target domain
    Test on test set of target domain
    '''
    args.work_dir = f'adapt_result/{args.dataset}_{args.source_type}/{args.adapt}/source{args.source}_target{args.target}'
    os.makedirs(args.work_dir,exist_ok=True)
    with open(os.path.join(args.work_dir,'config.json'),'w') as f:
        json.dump(args.__dict__,f,indent=2)
    # Set seed
    utils.set_seed(args.seed)
    
    # load target dataset
    assert args.target is not None
    adapt_dataset, test_dataset, target_domain = get_target_dataset(args)
    
    num_classes = len(test_dataset.underlying_dataset.classes)
    
    adapt_loader = InfiniteDataLoader(adapt_dataset,batch_size=args.batch_size,num_workers=4)
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size*3,shuffle=False,num_workers=4)
    
    # load model
    if args.dataset != 'Imagenet-C':
        model = ERM(num_classes).cuda()
        ckpt = torch.load(ckpt_path)['ckpt']
        model.load_state_dict(ckpt)
    else:
        model = torchvision.models.resnet50(pretrained=True).cuda()
    
    criterion = nn.CrossEntropyLoss()

    if args.adapt == 'stat' or args.adapt is None:
        optimizer = None
    else:
        bn_params = utils.get_bn_params(model,affine=args.adapt)
        optimizer = torch.optim.Adam(bn_params,lr=args.lr, weight_decay=args.wd)
    
    # Adapt to target data
    if args.adapt is None:
        freeze_option(model,args.adapt)
        test_loss,test_acc = test(model,criterion,test_loader)
        print(f"No adapt  Test Loss : {test_loss:.6f} Test Acc : {test_acc:.6f}")
        result = {'test_acc':test_acc}
        best_acc = test_acc
    else:
        adapt_iterator = iter(adapt_loader)
        adapt_loss = utils.AverageMeter()
        adapt_acc = utils.AverageMeter()
        best_acc = 0
        for step in range(args.adapt_step):            
            x,y = next(adapt_iterator)
            loss,acc = adapt(model,criterion,optimizer,x,y)
            adapt_loss.update(loss,x.size(0))
            adapt_acc.update(acc[0].item(),x.size(0))    

            # Logging & Test
            if step % args.val_freq == 0 or step == args.adapt_step-1:
                result = {
                    'adapt_step' : step,
                    'adapt_loss' : adapt_loss.avg,
                    'adapt_acc' : adapt_acc.avg,
                }
                adapt_loss.reset()
                adapt_acc.reset()
                
                # Test on target test set
                test_loss,test_acc = test(model,criterion,test_loader)
                result['test_loss'] = test_loss
                result['test_acc'] = test_acc
                
                if step == 0:
                    utils.print_row([k for k,v in result.items()],colwidth=12)
                utils.print_row([v for k,v in result.items()],colwidth=12)
                
                with open(os.path.join(args.work_dir,'adapt_log.json'),'a') as f:
                    f.write(json.dumps(result,sort_keys=True)+"\n")
                
                # Best acc
                if test_acc > best_acc:
                    best_acc = test_acc

    # inspect
    if args.dataset == 'Imagenet-C':
        init_ckpt = torchvision.models.resnet50(pretrained=True).state_dict()
        adapted_ckpt = model.to('cpu').state_dict()
    else:
        init_ckpt = torch.load(ckpt_path)['ckpt']
        adapted_ckpt = model.state_dict()
    updated_param = []

    for k,v in init_ckpt.items():
        if not torch.allclose(v,adapted_ckpt[k]):
            updated_param.append(k)
    if len(updated_param) == 0:
        print("No updated parameter")  
    else:
        should_be_0 = []
        for param in updated_param:
            if 'bn' not in param and 'downsample.1' not in param:
                should_be_0.append(param)
            if args.adapt == 'stat':
                if 'running_mean' in param:
                    continue
                if 'running_var' in param:
                    continue
                if 'num_batches_tracked' in param:
                    continue
            elif args.adapt == 'affine-scale':
                if 'weight' in param:
                    continue
            elif args.adapt == 'affine-shift':
                if 'bias' in param:
                    continue
            elif args.adapt == 'affine':
                if 'weight' in param or 'bias' in param:
                    continue
            elif args.adapt == 'BN':
                if 'running_mean' in param:
                    continue
                if 'running_var' in param:
                    continue
                if 'num_batches_tracked' in param:
                    continue
                if 'weight' in param or 'bias' in param:
                    continue
            should_be_0.append(param)
        if len(should_be_0) != 0:
            print(should_be_0)
    
    return best_acc, result['test_acc']

if __name__ == '__main__':
    if args.ckpt is not None:
        adapt_to_target(args.ckpt)
    else:
        num_domain = 4 if args.dataset in ['PACS','office_home','terra_incognita','VLCS'] else 6
        if args.dataset == 'Imagenet-C':
            num_domain = 15
            args.source_type = 'multi'
        source_type = args.source_type    
        dir_name = {'single':'source','multi':'target'}
        last_target_acc = {}
        best_target_acc = {}
        if args.target is not None:
            targets = [args.target]
        else:
            targets = list(range(num_domain))
        
        path = None
        for target in targets:
            if source_type == 'multi':
                args.target = target
                if args.dataset != 'Imagenet-C':
                    path = f"/nas/datahub/SFDA/ckpts/{args.dataset}_{source_type}_source/{dir_name[source_type]}_{target}"
                    path = os.path.join(path,os.listdir(path)[0],'ckpt.pth')
                
                source = list(range(num_domain))
                args.source = [str(d) for d in source if d!= args.target]
                args.source = ''.join(args.source)
                print(f"{args.dataset} {source_type} source {args.source} -> target {args.target}")
                best_acc,last_acc = adapt_to_target(path)
                last_target_acc[f'source{target}@target{target}'] = last_acc
                best_target_acc[f'source{target}@target{target}'] = best_acc
            else:
                for domain in range(num_domain):
                    if domain == target:
                        continue
                    if args.dataset != 'Imagenet-C':
                        args.target = target
                        path = f"/nas/datahub/SFDA/ckpts/{args.dataset}_{source_type}_source/{dir_name[source_type]}_{domain}"
                        path = os.path.join(path,os.listdir(path)[0],'ckpt.pth')
                    args.source = domain
                    print(f"{args.dataset} {source_type} source {args.source} -> target {target}")
                    best_acc,last_acc = adapt_to_target(path)
                    last_target_acc[f'source{target}@target{target}'] = last_acc
                    best_target_acc[f'source{target}@target{target}'] = best_acc
        
        best_target_acc = OrderedDict(sorted(best_target_acc.items()))
        last_target_acc = OrderedDict(sorted(last_target_acc.items()))
        print(f"Best target acc : \n{best_target_acc}")
        print(f"Last target acc : \n{last_target_acc}")
