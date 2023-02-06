import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transform = None
    def __getitem__(self, key):
        x,y = self.underlying_dataset[self.keys[key]]
        if self.transform is not None:
            x = self.transform(x)
        return x,y
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, ratio=0.2, seed=0):
    assert(ratio <= 0.5)
    keys = list(range(len(dataset)))
    classes = dataset.targets
    keys_1,keys_2 = train_test_split(keys,test_size=ratio,random_state=seed,stratify=classes)
    return keys_1, keys_2

def get_transform(mode,type_='default'):
    # TODO : define different type of augmentation
    if mode in ['val','test']:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    elif mode == 'train':
        if type_ == 'default':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif 'randaug' in type_:
            # type_ should be form of 'randaug_numops_magnitude'
            _, num_ops,magnitude = type_.split('_')
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224,scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=int(num_ops),magnitude=int(magnitude)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            raise ValueError("Wrong transform type")    
    else:
        raise ValueError("Wrong transform mode")
        
    return transform

def few_shot_subset(targets,n_shot):
    '''
    targets : torch.tensor contains labels of each data points
    n_shot : number of data point for each class
    Returns list contains indices of n_shot dataset
    '''
    class_, counts = torch.unique(targets,return_counts=True)
    indices = []
    val_indices = []
    for i, count in enumerate(counts):
        idx = torch.where(targets==class_[i])[0]
        if count < n_shot+1:
            raise ValueError(f"Class {class_[i]} only have {count} samples, {n_shot}-Shot is not available")
        else:
            temp = torch.randperm(len(idx))
            trn_idx, val_idx = idx[temp[:n_shot]], idx[temp[-1]]
        indices.extend(trn_idx.tolist())
        val_indices.append(val_idx)
    
    return indices, val_indices
    
def get_target_dataset(args):
    data_path = os.path.join(args.data_dir,args.dataset)
    domains = sorted([f.name for f in os.scandir(data_path) if f.is_dir()])

    tgt_domain = domains[args.target]
    print(tgt_domain)
     
    path = data_path + f'/{tgt_domain}'

    target_dataset = ImageFolder(os.path.join(path))
    tr_idx, te_idx = split_dataset(target_dataset, ratio=args.ratio, seed=args.seed)
    train_dataset, test_dataset = _SplitDataset(target_dataset, tr_idx), _SplitDataset(target_dataset, te_idx)

    train_dataset.transform = get_transform(mode='train',type_=args.aug)
    test_dataset.transform = get_transform(mode='test')

    if args.few_shot:
        keys = list(range(len(train_dataset)))
        targets = torch.tensor(train_dataset.underlying_dataset.targets)[train_dataset.keys]
        assert len(keys) == len(targets)
        
        few_shot_idx, val_idx = few_shot_subset(targets,args.few_shot) 
        
        # val_dataset = torch.utils.data.Subset(train_dataset,val_idx)
        # tr_name = [target_dataset.samples[i][0] for i in tr_idx]
        # val_name = [tr_name[i] for i in val_idx]
        # import pdb;pdb.set_trace()

        val_dataset = ImageFolder(os.path.join(f'./valset_cor_{args.dataset}/{args.seed}/{tgt_domain}'), transform=get_transform(mode='test'))
        adapt_dataset = torch.utils.data.Subset(train_dataset,few_shot_idx)
        
        # inspect few shot
        temp = []
        for _, y in adapt_dataset:
            temp.append(y)
        _, counts = torch.unique(torch.tensor(temp), return_counts=True)
        assert torch.allclose(counts.float(), torch.ones(max(targets)+1)*args.few_shot)
    else:
        adapt_dataset = train_dataset
    return adapt_dataset, val_dataset, test_dataset, tgt_domain
