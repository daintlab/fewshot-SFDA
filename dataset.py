import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
class SourceDomainImageFolder:
    '''
    Modified https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py
    src requires list of source domains in str type
    '''
    def __init__(self,root,src):
        super().__init__()
        assert type(src) == list
        self.environments = src        
        self.datasets = []
        for environment in self.environments:
            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path)

            self.datasets.append(env_dataset)
        
        self.num_classes = len(self.datasets[-1].classes)
        
class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transform = None
    def __getitem__(self, key):
        x,y = self.underlying_dataset[self.keys[key]]
        x = self.transform(x)
        return x,y
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, ratio=0.2, seed=0):
    assert(ratio <= 0.5)
    keys = list(range(len(dataset)))
    classes = dataset.targets
    keys_1,keys_2 = train_test_split(keys,test_size=ratio,random_state=seed,stratify=classes)
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

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
    for i, count in enumerate(counts):
        if count < n_shot:
            raise ValueError(f"Class {class_[i]} only have {count} samples, {n_shot}-Shot is not available")
        # TODO : n_shot보다 적은 데이터수를 가진 클래스 어떻게 처리할지

        idx = torch.where(targets==class_[i])[0]
        idx = idx[torch.randperm(len(idx))[:n_shot]]
        indices.extend(idx.tolist())
    
    return indices
    

def get_dataset(args):
    '''
    Returns source train/test datasets
    '''   
    data_path = os.path.join(args.data_dir,args.dataset)
    domains = sorted([f.name for f in os.scandir(data_path) if f.is_dir()])

    tgt_domain = domains[args.target]
    src_domains = [d for d in domains if d != tgt_domain] if args.source is None else [domains[args.source]]
    
    src_datasets = SourceDomainImageFolder(data_path,src_domains)
    num_classes = src_datasets.num_classes
    
    src_trn_datasets = []
    src_tst_datasets = []
    for dataset in src_datasets.datasets:
        big_split,small_split = split_dataset(dataset,ratio=args.ratio,seed=args.seed)
        big_split.transform = get_transform(mode='train',type_='default')
        small_split.transform = get_transform(mode='test')
        src_trn_datasets.append(big_split)
        src_tst_datasets.append(small_split)
        
    return src_trn_datasets,src_tst_datasets,src_domains,num_classes

    
def get_target_dataset(args):
    if args.dataset == 'Imagenet-C':
        data_path = args.data_dir+'/'+args.dataset
        domains = []
        for f in os.scandir(data_path):
            if f.is_dir() and f.name != 'extra':
                for d in os.scandir(data_path+'/'+f.name):
                    if d.is_dir():
                        domains.append(f.name+'/'+d.name)
        domains.sort()
    else:
        data_path = os.path.join(args.data_dir,args.dataset)
        domains = sorted([f.name for f in os.scandir(data_path) if f.is_dir()])

    tgt_domain = domains[args.target]
    print(tgt_domain)    
     
    if args.dataset == 'Imagenet-C':
        path = data_path + f'/{tgt_domain}/{args.severity}'
    else:
        path = data_path + f'/{tgt_domain}'

    target_dataset = ImageFolder(os.path.join(path))
    train_dataset, test_dataset = split_dataset(target_dataset, ratio=args.ratio, seed=args.seed)

    train_dataset.transform = get_transform(mode='train',type_=args.aug)
    test_dataset.transform = get_transform(mode='test')
        
    if args.few_shot:
        keys = list(range(len(train_dataset)))
        targets = torch.tensor(train_dataset.underlying_dataset.targets)[train_dataset.keys]
        assert len(keys) == len(targets)
        # _,few_shot_idx = train_test_split(
        #     keys,
        #     test_size=len(target_dataset.classes)*args.few_shot,
        #     random_state=args.seed,
        #     stratify=targets
        # )
        '''
        train_test_split does not guarantee {args.few_shot} sample per class
        '''
        few_shot_idx = few_shot_subset(targets,args.few_shot) 
        
        adapt_dataset = torch.utils.data.Subset(train_dataset,few_shot_idx)
        
        # inspect few shot
        temp = []
        for _,y in adapt_dataset:
            temp.append(y)
        _,counts = torch.unique(torch.tensor(temp),return_counts=True)
        assert torch.allclose(counts.float(),torch.ones(max(targets)+1)*args.few_shot)
    else:
        adapt_dataset = train_dataset

    return adapt_dataset,test_dataset,tgt_domain
    
    
