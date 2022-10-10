import argparse
import os
import json
import time
import torch
import torch.nn as nn
import numpy as np

from dataset import get_dataset
from data_loader import InfiniteDataLoader,FastDataLoader
from model import ERM
import utils


parser = argparse.ArgumentParser(description='Train on source domain data')
parser.add_argument('--data_dir', default='/data/domainbed',type=str,
                    help='Data directory')
parser.add_argument('--dataset', default='PACS',type=str,
                    help='Data directory')
parser.add_argument('--target', default=None,type=int,
                    help='Index of target domain')
parser.add_argument('--source', default=None,type=int,
                    help='Index of source domain. None for multi-source training')
parser.add_argument('--save_path', default='./result',type=str,
                    help='Save path')
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
parser.add_argument('--steps', default=5000,type=int,
                    help='Training step')
parser.add_argument('--val_freq', default=200,type=int,
                    help='Validation frequency')
args = parser.parse_args()


def validate(model,criterion,loader):
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
    model.train()
    return test_loss.avg, test_acc.avg
            

def main():    
    # Set seed
    utils.set_seed(args.seed)

    # save config
    args.work_dir = os.path.join(args.save_path,utils.timestamp())
    os.makedirs(args.work_dir,exist_ok=True)
    with open(os.path.join(args.work_dir,'config.json'),'w') as f:
        json.dump(args.__dict__,f,indent=2)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # load data
    train_dataset,test_dataset,source_domains,num_classes = get_dataset(args)
    train_loaders = [InfiniteDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=4,
    ) for dataset in train_dataset]
    test_loaders = [FastDataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=4
    )for dataset in test_dataset]

    # create model
    model = ERM(num_classes=num_classes).cuda()

    # define loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)
    
    # train
    train_iterator = zip(*train_loaders)
    train_loss = utils.AverageMeter()
    iter_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    end = time.time()
    best_acc = 0
    for step in range(args.steps):
        minibatches = [(x.cuda(), y.cuda())
            for x,y in next(train_iterator)]
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        data_time.update(time.time()-end)
        
        output = model(all_x)
        loss = criterion(output,all_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(),all_x.size(0))
        iter_time.update(time.time()-end)
        end = time.time()
        
        if (step % args.val_freq == 0) or (step == args.steps-1):
            result = {'step':step,
                      'train_loss':train_loss.avg}
            train_loss.reset()
            
            # validate
            avg_acc = 0
            for i,loader in enumerate(test_loaders):
                loss,acc = validate(model,criterion,loader)
                result[f"test@src{i}_loss"] = loss
                result[f"test@src{i}_acc"] = acc
                avg_acc += acc
            avg_acc = avg_acc / len(test_loaders)
            result['test_avg_acc'] = avg_acc
            
            # log
            result['iter_time'] = iter_time.avg
            result['data_time'] = data_time.avg
            if step == 0:
                utils.print_row([k for k,v in result.items()],colwidth=16)
            utils.print_row([v for k,v in result.items()],colwidth=16)
            with open(os.path.join(args.work_dir,'log.json'),'a') as f:
                f.write(json.dumps(result,sort_keys=True)+"\n")
            
            # save model
            if avg_acc > best_acc:
                save_dict = {
                    'ckpt' : model.state_dict(),
                    'source_acc' : avg_acc,
                    'best_step' : step
                }
                torch.save(save_dict,os.path.join(args.work_dir,'ckpt.pth'))
                best_acc = avg_acc

    print(f"Best Source ACC : {best_acc:.4f} at step :{save_dict['best_step']}")
    final_result = {
        'source': source_domains,
        'best_source_acc' : best_acc,
        'best_step':save_dict['best_step']
    }
    with open(os.path.join(args.work_dir,'log.json'),'a') as f:
        f.write(json.dumps(final_result,sort_keys=True)+"\n")

if __name__ == '__main__':
    main()