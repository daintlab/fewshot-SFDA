import argparse
import os
import json
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
from sam import SAM
from model import SHOT,SHOT_fe,IMGNET

from dataset import get_target_dataset 
from data_loader import InfiniteDataLoader
import utils
import warnings
from sklearn.metrics import confusion_matrix
import pickle
warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser(description='Adapt and test on target domain')
parser.add_argument('--data_dir', default='/data/domainbed',type=str,help='Data directory')
parser.add_argument('--dataset', default='office_home',type=str,
                    choices=['office_home','terra_incognita','VLCS','office31','VISDA-C','office_home_RSUT','VISDA-C_RSUT'],
                    help='Data directory')
parser.add_argument('--work_dir', default='./result',type=str,
                    help='Working directory')
parser.add_argument('--source', default=None,type=int,
                    help='Index of source domain')
# parser.add_argument('--target', default=None,type=int,help='Index of target domain')
parser.add_argument('--pretrain', default='SHOT',type=str,
                    choices=['SHOT','SHOT_LP','IMGNET','IMGNET_LP'],help='Source pretrain strategy')
parser.add_argument('--adapt', default=None,type=str,
                    choices=['cls','clsBN','BN','feat','all'],help='Fine-tuning method')
parser.add_argument('--adapt_step', default=1000,type=int,
                    help='Adaptation step')
parser.add_argument('--batch_size', default=32,type=int,
                    help='Batch size per domain')
parser.add_argument('--optim', default='adam',type=str,
                    choices=['adam','sgd'],help='Optimizer for adaptation')
parser.add_argument('--lr', default=0.001,type=float,
                    help='Learning rate')
parser.add_argument('--wd', default=0,type=float,
                    help='Weight decay')
parser.add_argument('--ratio', default=0.2,type=float,
                    help='Holdout ratio for target test set')
parser.add_argument('--val_freq', default=100,type=int,
                    help='Evaluation frequency')
parser.add_argument('--few_shot', default=None,type=int,
                    help='adapt for a few images')
parser.add_argument('--SAM', action='store_true',default=False,
                    help='Use Sharpness aware minimization')
parser.add_argument('--aug', default='default',type=str, 
                    choices=['default','randaug'],help='Type of augmentation for training stage')
parser.add_argument('--rho', default=0.05,type=float,
                    help='SAM rho')
parser.add_argument('--seed', default=0,type=int,
                    help='Random seed')
parser.add_argument('--ckpt_dir', default=None,type=str, 
                    help='Path to source pretrained models directory')
parser.add_argument('--label_smoothing', default=0,type=float,
                    help='label smoothing parameter')
parser.add_argument('--oda_seed', default=2020,type=int,
                    help='Known class selection seed for OoD scenario')
parser.add_argument('--subset',action='store_true',default=False,
                    help='True for OoD scenario')
parser.add_argument('--imbalance',action='store_true',default=False,
                    help='True for Imbalance scenario')
args = parser.parse_args()

def train_one_step(model,criterion,optimizer,x,y,args):
    utils.freeze_proper_param(model,args)
    x,y = x.cuda(),y.cuda()
    
    output = model(x)
    loss = criterion(output,y)
    
    if args.SAM:
        loss.backward()
        optimizer.first_step(zero_grad=True)
        
        criterion(model(x), y).backward()
        optimizer.second_step(zero_grad=True)
    else:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    acc,_= utils.accuracy(output, y, topk=(1,5))
    return loss.item(), acc

def test(model,criterion,loader,flag=False):
    model.eval()
    test_loss = utils.AverageMeter()
    test_acc = utils.AverageMeter()
    with torch.no_grad():
        for i,(x,y) in enumerate(loader):
            x,y = x.cuda(), y.cuda()
            output = model(x)
            loss = criterion(output, y)
            if flag:
                if i == 0:
                    total_pred = torch.argmax(output.detach().cpu(), 1)
                    total_label = y.detach().cpu()
                else:
                    total_pred = torch.cat((total_pred, torch.argmax(output.detach().cpu(), 1)))
                    total_label = torch.cat((total_label, y.detach().cpu()))
            acc,_= utils.accuracy(output, y, topk=(1,5))
            test_acc.update(acc[0].item(), x.size(0))
            test_loss.update(loss.item(), x.size(0))
    if flag:
        cm = confusion_matrix(total_label, total_pred)
        per_class_acc = cm.diagonal()/cm.sum(axis=1) * 100
        per_class_acc = per_class_acc.mean()            
        return test_loss.avg, test_acc.avg, per_class_acc
    else:
        return test_loss.avg, test_acc.avg

def train_on_target(args):
    save_path = os.path.join(args.work_dir, f'target{args.target}')
    os.makedirs(save_path, exist_ok=True)
    
    # data
    train_dataset, val_dataset, test_dataset, target_domain = get_target_dataset(args)
    
    # Define number of classes
    if args.subset: # OoD scenario
        if args.dataset == 'office_home':
            num_classes = 25
        if args.dataset == 'office31':
            num_classes = 10
        if args.dataset == 'VISDA-C':
            num_classes = 6
    elif args.imbalance:
        if args.dataset == 'office_home':
            num_classes = 65
        if args.dataset == 'VISDA-C':
            num_classes = 12
    else:
        num_classes = len(test_dataset.underlying_dataset.classes)

    # Handle train data < batch size
    if len(train_dataset) < args.batch_size:
        multiplier = int(args.batch_size / len(train_dataset)) + 1
        train_dataset.indices = train_dataset.indices * multiplier
    
    train_loader = InfiniteDataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=8
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size*3, shuffle=False, num_workers=8
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size*3, shuffle=False, num_workers=8
    )
    
    # model
    if args.pretrain == 'SHOT':
        model = SHOT(ckpts=args.ckpts, dataset=args.dataset, subset=args.subset).cuda()
    elif args.pretrain == 'SHOT_LP':
        model = SHOT_fe(ckpts=args.ckpts, dataset=args.dataset, subset=args.subset).cuda()
    elif args.pretrain == 'IMGNET' or args.pretrain == 'IMGNET_LP':
        model = IMGNET(ckpts=args.ckpts, dataset=args.dataset).cuda()
    else:
        raise NotImplementedError
    
    # Define loss
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Define optimizer
    if args.adapt is not None:
        param = utils.get_proper_param(model,args)
        if args.SAM:
            if args.optim == 'adam':
                optimizer = SAM(param, torch.optim.Adam, lr=args.lr, weight_decay=args.wd, rho=args.rho)
            else:
                optimizer = SAM(param, torch.optim.SGD, lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True, rho=args.rho)
        else:
            if args.optim == 'adam':
                optimizer = torch.optim.Adam(param, lr=args.lr, weight_decay=args.wd)
            else:
                optimizer = torch.optim.SGD(param, lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True)
        
    # train
    train_iterator = iter(train_loader)
    val_iterator = iter(val_loader)
    train_loss = utils.AverageMeter()
    train_acc = utils.AverageMeter()
    best_acc = 0
    best_val_loss = 1000
    seleted_acc = 0

    if args.adapt is None:
        test_loss,test_acc = test(model,criterion,test_loader)
        print(f"No adapt  Test Loss : {test_loss:.6f} Test Acc : {test_acc:.6f}")
        result = {'test_acc':test_acc}
    else:
        for step in range(args.adapt_step):
            x,y = next(train_iterator)
            loss,acc = train_one_step(model,criterion,optimizer,x,y,args)
            train_loss.update(loss,x.size(0))
            train_acc.update(acc[0].item(),x.size(0))

            # Logging & Test
            if step % args.val_freq == 0 or step == args.adapt_step-1:
                result = {
                    'train_step' : step,
                    'train_loss' : train_loss.avg,
                    'train_acc' : train_acc.avg,
                }
                train_loss.reset()
                train_acc.reset()
                
                # Validate on target validation set
                val_loss,val_acc = test(model,criterion,val_loader)
                result['val_loss'] = val_loss
                result['val_acc'] = val_acc
                
                # Test on target test set
                if args.dataset in ['VISDA-C','VLCS','terra_incognita'] or args.imbalance:
                    test_loss,test_acc,test_per_class_acc = test(model,criterion,test_loader,flag=True)
                else:
                    test_loss,test_acc = test(model,criterion,test_loader)
                    test_per_class_acc = 0.0
                result['test_loss'] = test_loss
                result['test_acc'] = test_acc
                result['test_perclass'] = test_per_class_acc
                
                if step == 0:
                    utils.print_row([k for k,v in result.items()],colwidth=12)
                utils.print_row([v for k,v in result.items()],colwidth=12)
                
                with open(os.path.join(save_path,'train_log.json'),'a') as f:
                    f.write(json.dumps(result,sort_keys=True)+"\n")
                
                # Selected acc
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    seleted_acc = test_acc
                    if args.dataset in ['VISDA-C','VLCS','terra_incognita'] or args.imbalance:
                        selected_per_class_acc = test_per_class_acc
                    torch.save(model.state_dict(),os.path.join(save_path,'val_best_ckpt.pth'))
                    
                # Best acc
                if test_acc > best_acc:
                    best_acc = test_acc
                    if args.dataset in ['VISDA-C','VLCS','terra_incognita'] or args.imbalance:
                        best_per_class_acc = test_per_class_acc
                    torch.save(model.state_dict(),os.path.join(save_path,'test_best_ckpt.pth'))
        # Model save
        torch.save(model.state_dict(),os.path.join(save_path,'last_ckpt.pth'))
    
    if args.dataset in ['VISDA-C','VLCS','terra_incognita'] or args.imbalance:
        return best_acc, seleted_acc, result['test_acc'], best_per_class_acc, selected_per_class_acc, test_per_class_acc
    else:
        return best_acc, seleted_acc, result['test_acc']

def get_ckpt(args):
    domain_dict = {
                'office_home' : ['A','C','P','R'],
                'PACS' : ['A','C','P','S'],
                'VLCS' : ['C','L','S','V'],
                'office31' : ['A','D','W'],
                'terra_incognita' : ['L100','L38','L43','L46'],
                'VISDA-C' : ['T','V'],
                'office_home_RSUT' : ['C','P','R'],
                'VISDA-C_RSUT' : ['T','V'],
            }

    if args.pretrain == 'SHOT':
        root_dir = args.ckpt_dir
        
        ckpts = {}
        ckpts['netF'] = os.path.join(root_dir,f'{args.dataset}/{domain_dict[args.dataset][args.source]}/source_F.pt')
        ckpts['netB'] = os.path.join(root_dir,f'{args.dataset}/{domain_dict[args.dataset][args.source]}/source_B.pt')
        ckpts['netC'] = os.path.join(root_dir,f'{args.dataset}/{domain_dict[args.dataset][args.source]}/source_C.pt')
        
    elif args.pretrain == 'SHOT_LP' or args.pretrain == 'IMGNET_LP':
        root_dir = args.ckpt_dir
        ckpts = os.path.join(root_dir,f'source{args.source}/target{args.target}/last_ckpt.pth')
        print('Load LP model')
        
    elif args.pretrain == 'IMGNET':
        ckpts = None
        print('Source model is pretrained with ImageNet')
    
    else:
        raise NotImplementedError        
  
    return ckpts

def main():
    # Save config
    args.work_dir = os.path.join(f'./logs/{args.dataset}',f'{args.work_dir}/source{args.source}')
    os.makedirs(args.work_dir,exist_ok=True)
    with open(os.path.join(args.work_dir,'config.json'),'w') as f:
        json.dump(args.__dict__,f,indent=2)
    
    # Set seed
    utils.set_seed(args.seed)

    # Accuracy
    last_target_acc = {}
    selected_target_acc = {}
    best_target_acc = {}
    # Per-class Accuracy
    last_target_p_acc = {}
    selected_target_p_acc = {}
    best_target_p_acc = {}
    
    if args.dataset == 'office31' or args.dataset == 'office_home_RSUT':
        num_d = 3
    elif args.dataset == 'VISDA-C' or args.dataset == 'VISDA-C_RSUT':
        num_d = 2
    else:
        num_d = 4
        
    # Adapt to all possible target domains from given source pretrained model
    for target in range(num_d):
        if target == args.source:
            continue
        args.target = target
        args.ckpts = get_ckpt(args)
        print(f"{args.dataset} source {args.source} -> target {target}")
        if args.dataset in ['VISDA-C','VLCS','terra_incognita'] or args.imbalance:
            per_class_acc = {}
            best_acc,selected_acc,last_acc,best_per_class_acc,selected_per_class_acc,last_per_class_acc = train_on_target(args)
        else:
            best_acc,selected_acc,last_acc = train_on_target(args)
        
        last_target_acc[f'source{args.source}@target{target}'] = last_acc
        selected_target_acc[f'source{args.source}@target{target}'] = selected_acc
        best_target_acc[f'source{args.source}@target{target}'] = best_acc
        
        if args.dataset in ['VISDA-C','VLCS','terra_incognita'] or args.imbalance:
            last_target_p_acc[f'source{args.source}@target{target}'] = last_per_class_acc
            selected_target_p_acc[f'source{args.source}@target{target}'] = selected_per_class_acc
            best_target_p_acc[f'source{args.source}@target{target}'] = best_per_class_acc

    # Aggregate Results
    best_target_acc = OrderedDict(sorted(best_target_acc.items()))
    selected_target_acc = OrderedDict(sorted(selected_target_acc.items()))
    last_target_acc = OrderedDict(sorted(last_target_acc.items()))
    with open(os.path.join(args.work_dir,'best_target.json'),'w') as f:
        json.dump(best_target_acc,f)
    with open(os.path.join(args.work_dir,'selected_target.json'),'w') as f:
        json.dump(selected_target_acc,f)
    with open(os.path.join(args.work_dir,'last_target.json'),'w') as f:
        json.dump(last_target_acc,f)
    print(f"Best target acc : \n{best_target_acc}")
    print(f"Selected target acc : \n{selected_target_acc}")
    print(f"Last target acc : \n{last_target_acc}")
    
    if args.dataset in ['VISDA-C','VLCS','terra_incognita'] or args.imbalance:
        best_target_p_acc = OrderedDict(sorted(best_target_p_acc.items()))
        selected_target_p_acc = OrderedDict(sorted(selected_target_p_acc.items()))
        last_target_p_acc = OrderedDict(sorted(last_target_p_acc.items()))
        with open(os.path.join(args.work_dir,'best_target.json'),'w') as f:
            json.dump(best_target_p_acc,f)
        with open(os.path.join(args.work_dir,'selected_target.json'),'w') as f:
            json.dump(selected_target_p_acc,f)
        with open(os.path.join(args.work_dir,'last_target.json'),'w') as f:
            json.dump(last_target_p_acc,f)
        print(f"Best target Per-class acc : \n{best_target_p_acc}")
        print(f"Selected target Per-class acc : \n{selected_target_p_acc}")
        print(f"Last target Per-class acc : \n{last_target_p_acc}")
    

if __name__ == '__main__':
    main()
