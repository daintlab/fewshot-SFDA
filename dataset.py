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
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    src_trn_datasets = []
    src_tst_datasets = []
    for dataset in src_datasets.datasets:
        big_split,small_split = split_dataset(dataset,ratio=args.ratio,seed=args.seed)
        big_split.transform = train_transform
        small_split.transform = val_transform
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
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    if args.dataset == 'Imagenet-C':
        target_dataset = ImageFolder(os.path.join(data_path,tgt_domain+f'/{args.severity}'),transform=val_transform)
    else:
        target_dataset = ImageFolder(os.path.join(data_path,tgt_domain),transform=val_transform)
    
    if args.few_shot is not None:
        _, test_ind, _, _ = train_test_split(range(len(target_dataset)),target_dataset.targets,
        stratify=target_dataset.targets,test_size=len(target_dataset.classes)*args.few_shot,random_state=1)
        target_subset = torch.utils.data.Subset(target_dataset,test_ind)
        return target_dataset,target_subset,tgt_domain

    return target_dataset,tgt_domain
    
    
