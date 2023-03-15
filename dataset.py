import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from PIL import ImageFile, Image
import numpy as np
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

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList_idx(torch.utils.data.Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self):
        return len(self.imgs)

def split_dataset(dataset, args, ratio=0.2, seed=0):
    assert(ratio <= 0.5)
    keys = list(range(len(dataset)))
    if args.subset or args.imbalance:
        classes = [i[1] for i in dataset.imgs]
    else:
        classes = dataset.targets
    keys_1,keys_2 = train_test_split(keys,test_size=ratio,random_state=seed,stratify=classes)

    return keys_1, keys_2

def get_transform(mode,type_='default'):
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
    if args.subset:
        domains = sorted([f.name.split('_')[0] for f in os.scandir(data_path)])
    elif args.imbalance:
        domains = sorted(set([f.name.split('_')[0] for f in os.scandir(data_path)]))
    else:
        domains = sorted([f.name for f in os.scandir(data_path) if f.is_dir()])

    tgt_domain = domains[args.target]
    print(f"Target Domain : {tgt_domain}")
     
    path = data_path + f'/{tgt_domain}'

    if args.subset:
        if args.dataset == 'office_home':
            args.class_num = 25
            known_class = np.random.RandomState(seed=args.oda_seed).permutation(65)[:args.class_num]
        if args.dataset == 'office31':
            args.class_num = 10
            known_class = np.random.RandomState(seed=args.oda_seed).permutation(31)[:args.class_num]
        if args.dataset == 'VISDA-C':
            args.class_num = 6
            known_class = np.random.RandomState(seed=args.oda_seed).permutation(12)[:args.class_num]
        txt_tar = open(f'{args.data_dir}/{args.dataset}/{tgt_domain}_list.txt').readlines()
        label_dict = {known_class[i]:i for i in range(len(known_class))}
        known_tar = []
        t_classes = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in known_class:
                line = reci[0] + ' ' + str(label_dict[int(reci[1])]) + '\n'   
                known_tar.append(line)
                t_classes.append(label_dict[int(reci[1])])
        txt_known = known_tar.copy()
        target_dataset = ImageList_idx(txt_known)
    elif args.imbalance:
        txt_tar = open(f'{args.data_dir}/{args.dataset}/{tgt_domain}_UT.txt').readlines()
        target_dataset = ImageList_idx(txt_tar)
    else:
        target_dataset = ImageFolder(os.path.join(path))

    tr_idx, te_idx = split_dataset(target_dataset, args, ratio=args.ratio, seed=args.seed)
    train_dataset, test_dataset = _SplitDataset(target_dataset, tr_idx), _SplitDataset(target_dataset, te_idx)

    train_dataset.transform = get_transform(mode='train',type_=args.aug)
    test_dataset.transform = get_transform(mode='test')

    if args.few_shot:
        keys = list(range(len(train_dataset)))
        if args.subset or args.imbalance:
            targets = torch.tensor([target_dataset.imgs[i][1] for i in tr_idx])
        else:
            targets = torch.tensor(train_dataset.underlying_dataset.targets)[train_dataset.keys]
        assert len(keys) == len(targets)
        few_shot_idx, val_idx = few_shot_subset(targets,args.few_shot) 

        val_dataset = torch.utils.data.Subset(train_dataset,val_idx)
        adapt_dataset = torch.utils.data.Subset(train_dataset,few_shot_idx)
        
        # inspect few shot
        temp = []
        for _,y in adapt_dataset:
            temp.append(y)
        _,counts = torch.unique(torch.tensor(temp),return_counts=True)
        if args.subset:
            assert torch.allclose(counts.float(),torch.ones(args.class_num)*args.few_shot)
        else:
            assert torch.allclose(counts.float(),torch.ones(max(targets)+1)*args.few_shot)
    else:
        adapt_dataset = train_dataset
    return adapt_dataset,val_dataset,test_dataset,tgt_domain