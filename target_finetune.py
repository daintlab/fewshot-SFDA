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
import matplotlib.pyplot as plt
from model import ERM,SHOT
from dataset2 import get_target_dataset
from dataset import get_target_dataset as get_target_dataset_robust
from data_loader import InfiniteDataLoader
import utils
import warnings

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser(description='Adapt and test on target domain')
parser.add_argument('--data_dir', default='/data/domainbed',type=str,help='Data directory')
parser.add_argument('--dataset', default='PACS',type=str,
                    choices=['PACS','office_home','terra_incognita','VLCS','office31'],help='Data directory')
parser.add_argument('--work_dir', default='./result',type=str,help='Working directory')
parser.add_argument('--source', default=None,type=int,help='Index of source domain')
parser.add_argument('--target', default=None,type=int,help='Index of target domain')
parser.add_argument('--pretrain', default='SHOT',type=str,choices=['SHOT'],help='Source pretrain strategy')

parser.add_argument('--adapt', default=None,type=str,choices=['cls','clsBN','BN','feat','all'],help='Fine-tuning method')
parser.add_argument('--adapt_step', default=1000,type=int,help='Adaptation step')
parser.add_argument('--batch_size', default=32,type=int,help='Batch size per domain')
parser.add_argument('--optim', default='adam',type=str,choices=['adam','sgd'],help='Optimizer for adaptation')
parser.add_argument('--lr', default=0.001,type=float,help='Learning rate')
parser.add_argument('--wd', default=0,type=float,help='Weight decay')

parser.add_argument('--ratio', default=0.2,type=float,help='Holdout ratio (validation)')
parser.add_argument('--val_freq', default=100,type=int,help='Validation frequency')
parser.add_argument('--few_shot', default=None,type=int,help='adapt for a few images')

parser.add_argument('--SAM', action='store_true',default=False,help='Use SAM')
parser.add_argument('--mixup', default=None,type=float,help='alpha value of beta distribution when use mixup')
parser.add_argument('--regmixup', action='store_true',default=False,help='whether to use regmixup')
parser.add_argument('--label_smoothing', default=0,type=float,help='label smoothing parameter')
parser.add_argument('--aug', default='default',type=str, 
                    choices=['default','randaug'],help='Type of augmentation for training stage')
parser.add_argument('--rho', default=0.05,type=float,help='SAM rho')
parser.add_argument('--flood', type=float,default=None,help='loss flooding')
parser.add_argument('--robust',action='store_true',default=False,help='Robust validation')
parser.add_argument('--seed', default=0,type=int,help='Random seed')
args = parser.parse_args()

def train_one_step(model,criterion,optimizer,x,y,args):
    utils.freeze_proper_param(model,args)
    x,y = x.cuda(),y.cuda()
    
    # mixup
    if args.mixup:
        mixed_x, y_a, y_b, lambd = utils.mixup_data(x, y, alpha=args.mixup)
        if args.regmixup:
            targets_a = torch.cat([y, y_a])
            targets_b = torch.cat([y, y_b])
            x = torch.cat([x, mixed_x], dim=0)
            logits = model(x)
            output = logits[:len(logits)//2].detach()
            loss = utils.mixup_criterion(criterion, logits, targets_a, targets_b, lambd)
        else:
            output = model(mixed_x)
            loss = utils.mixup_criterion(criterion, mixed_output, y_a, y_b, lambd)
    else:
        output = model(x)
        loss = criterion(output,y)
    
    if args.SAM:
        loss.backward()
        optimizer.first_step(zero_grad=True)
        if args.flood:
            ((criterion(model(x),y) - args.flood).abs() + args.flood).backward()
        elif args.mixup:
            if args.regmixup:
                utils.mixup_criterion(criterion, model(x), targets_a, targets_b, lambd).backward()
            else:
                utils.mixup_criterion(criterion, model(x), y_a, y_b, lambd).backward()
        else:
            criterion(model(x), y).backward()
        optimizer.second_step(zero_grad=True)
    else:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc,_= utils.accuracy(output, y, topk=(1,5))
    return loss.item(), acc

def test(model,criterion,loader):
    model.eval()
    test_loss = utils.AverageMeter()
    test_acc = utils.AverageMeter()
    with torch.no_grad():
        for i,(x,y) in enumerate(loader):
            x,y = x.cuda(), y.cuda()
            output = model(x)
            loss = criterion(output, y)
            
            acc,_= utils.accuracy(output, y, topk=(1,5))
            test_loss.update(loss.item(), x.size(0))
            test_acc.update(acc[0].item(), x.size(0))
    
    return test_loss.avg, test_acc.avg

def train_on_target(args):
    save_path = os.path.join(args.work_dir, f'target{args.target}')
    os.makedirs(save_path, exist_ok=True)
    
    # data
    if args.robust:
        train_dataset, val_dataset, test_dataset, target_domain = get_target_dataset_robust(args)
    else:
        train_dataset, val_dataset, test_dataset, target_domain = get_target_dataset(args)
    num_classes = len(test_dataset.underlying_dataset.classes)

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
        model = SHOT(ckpts=args.ckpts, dataset=args.dataset).cuda()
    else:
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
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
    best_loss = 1000
    seleted_acc = 0

    loss_plot = []
    acc_plot = []

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
            loss_plot.append(loss)
            acc_plot.append(acc.cpu())
            
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
                test_loss,test_acc = test(model,criterion,test_loader)
                result['test_loss'] = test_loss
                result['test_acc'] = test_acc
                
                if step == 0:
                    utils.print_row([k for k,v in result.items()],colwidth=12)
                utils.print_row([v for k,v in result.items()],colwidth=12)
                
                with open(os.path.join(save_path,'train_log.json'),'a') as f:
                    f.write(json.dumps(result,sort_keys=True)+"\n")
                
                # Selected acc
                if best_loss > val_loss:
                    best_loss = val_loss
                    seleted_acc = test_acc

                # Best acc
                if test_acc > best_acc:
                    best_acc = test_acc
        
        # 충분히 수렴하는지 확인하기 위한 시각화 코드
        # plt.plot(loss_plot, label='loss')
        # plt.legend()
        # plt.show()
        # plt.savefig(f'./plot_{args.adapt}_{args.few_shot}shot_{args.source}_to_{args.target}_loss.png')
        # plt.close()
        # plt.plot(acc_plot, label='acc')
        # plt.legend()
        # plt.show()
        # plt.savefig(f'./plot_{args.adapt}_{args.few_shot}shot_{args.source}_to_{args.target}_acc.png')
        # plt.close()

        # Model save
        torch.save(model.to('cpu').state_dict(),os.path.join(save_path,'ckpt.pth'))
    
    return best_acc, seleted_acc, result['test_acc']

def get_ckpt(args):
    domain_dict = {
                'office_home' : ['A','C','P','R'],
                'PACS' : ['A','C','P','S'],
                'VLCS' : ['C','L','S','V'],
                'office31' : ['A','D','W']
            }

    if args.pretrain == 'SHOT':
        if args.adapt is not None:
            if args.seed == 0:
                fol_name = 'source_seed2020'
            elif args.seed == 1:
                fol_name = 'source_seed2021'
            elif args.seed == 2:
                fol_name = 'source_seed2022'
            
            root_dir = f'/nas/datahub/SFDA/shot_src_pretrained/ckps/{fol_name}/uda'
            
            ckpts = {}
            ckpts['netF'] = os.path.join(root_dir,f'{args.dataset}/{domain_dict[args.dataset][args.source]}/source_F.pt')
            ckpts['netB'] = os.path.join(root_dir,f'{args.dataset}/{domain_dict[args.dataset][args.source]}/source_B.pt')
            ckpts['netC'] = os.path.join(root_dir,f'{args.dataset}/{domain_dict[args.dataset][args.source]}/source_C.pt')
        
        # 여기는 특정 모델 Test하고 싶을 때 사용하는 코드, test하고 싶은 모델 load하는 부분이니까 바꿔사용하면됨
        # else:
        #     if args.seed == 0:
        #         fol_name = f'seed2020'
        #         seed = 2020
        #     elif args.seed == 1:
        #         fol_name = f'seed2021'
        #         seed = 2021
        #     elif args.seed == 2:
        #         fol_name = f'seed2022'
        #         seed = 2022
            
        #     root_dir = f'/nas/home/tmddnjs3467/domain-generalization/SHOT/object/ckps/target_samesplit_{fol_name}/uda'
        #     pair = domain_dict[args.dataset][args.source] + domain_dict[args.dataset][args.target]

        #     ckpts = {}
        #     ckpts['netF'] = os.path.join(root_dir,f'{args.dataset}/{pair}/target_F_par_0.3.pt')
        #     ckpts['netB'] = os.path.join(root_dir,f'{args.dataset}/{pair}/target_B_par_0.3.pt')
        #     ckpts['netC'] = os.path.join(root_dir,f'{args.dataset}/{pair}/target_C_par_0.3.pt')
        
        #     root_dir = f'/nas/home/tmddnjs3467/domain-generalization/CoWA-JMDS/ckps/source/uda'
        #     ckpts = {}
        #     ckpts['netF'] = os.path.join(root_dir,f'{args.dataset}/{domain_dict[args.dataset][args.source]}/source_F_{seed}.pt')
        #     ckpts['netB'] = os.path.join(root_dir,f'{args.dataset}/{domain_dict[args.dataset][args.source]}/source_B_{seed}.pt')
        #     ckpts['netC'] = os.path.join(root_dir,f'{args.dataset}/{domain_dict[args.dataset][args.source]}/source_C_{seed}.pt')
        
    else:
        raise NotImplementedError
    
    return ckpts

def main():
    # Save config
    args.work_dir = os.path.join(f'./D_fine_tuning_output/{args.dataset}',f'{args.work_dir}/source{args.source}')
    os.makedirs(args.work_dir,exist_ok=True)
    with open(os.path.join(args.work_dir,'config.json'),'w') as f:
        json.dump(args.__dict__,f,indent=2)
    
    # Set seed
    utils.set_seed(args.seed)

    assert args.dataset in ['PACS','office_home','terra_incognita','VLCS','office31']
    last_target_acc = {}
    selected_target_acc = {}
    best_target_acc = {}
    if args.dataset == 'office31':
        num_d = 3
    else:
        num_d = 4
    for target in range(num_d):
        if target == args.source :
            continue
        args.target = target
        args.ckpts = get_ckpt(args)
        print(f"{args.dataset} source {args.source} -> target {target}")
        best_acc,selected_acc,last_acc = train_on_target(args)
        
        last_target_acc[f'source{args.source}@target{target}'] = last_acc
        selected_target_acc[f'source{args.source}@target{target}'] = selected_acc
        best_target_acc[f'source{args.source}@target{target}'] = best_acc
    # 결과 aggregate
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
    

if __name__ == '__main__':
    main()
