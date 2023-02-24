import torch
import torch.nn.functional as F
from torch.autograd import Variable
import json

def entropy(p, prob=True, mean=True):
    if prob:
        p = F.softmax(p, dim=-1)
    en = -torch.sum(p * torch.log(p+1e-5), 1)
    if mean:
        return torch.mean(en)
    else:
        return en


def neighbor_density(feature, T=0.05):
    feature = F.normalize(feature)
    mat = torch.matmul(feature, feature.t()) / T
    mask = torch.eye(mat.size(0), mat.size(0)).bool()
    mat.masked_fill_(mask, -1 / T)
    result = entropy(mat)
    return result

def test_and_nd(step, dataset_test, name, netF, netB, netC):
    netF.eval()
    netB.eval()
    netC.eval()
    
    correct = 0
    size = 0
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t  = data[0], data[1]
            img_t, label_t = Variable(img_t.cuda()), \
                             Variable(label_t.cuda())
            feat = netB(netF(img_t))
            out_t = netC(feat).cpu()
            pred = out_t.data.max(1)[1]
            correct += pred.eq(label_t.data.cpu()).cpu().sum()
            k = label_t.data.size()[0]
            size += k
            if batch_idx == 0:
                label_all = label_t
                feat_all = feat
                pred_all = out_t
            else:
                pred_all = torch.cat([pred_all, out_t],0)
                feat_all = torch.cat([feat_all, feat],0)
                label_all = torch.cat([label_all, label_t],0)
    print(
        '\nTest set:  Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, size,
            100. * correct / size))
    ## Accuracy
    close_p = 100. * float(correct) / float(size)
    #compute_variance(pred_all, label_all)
    #compute_variance(feat_all, label_all)
    ## Entropy
    ent_class = entropy(pred_all)

    ## Neighborhood Density
    pred_soft = F.softmax(pred_all,dim=-1)
    nd_soft = neighbor_density(pred_soft)

    ## Neighborhood Density without softmax
    nd_nosoft = neighbor_density(pred_all)

    output = [step, "closed", "acc %s"%float(close_p),
              "neighborhood density %s"%nd_soft.item(),
              "neighborhood density no soft%s" % nd_nosoft.item(),
              "entropy class %s"%ent_class.item()]

    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=name, format="%(message)s")
    logger.setLevel(logging.INFO)
    print(output)
    logger.info(output)
    return close_p, nd_soft.item(), nd_nosoft.item(), ent_class.item()

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import ImageFile
import os 
from sklearn.model_selection import train_test_split
import network
from torch.utils.data import DataLoader
from itertools import permutations

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

ImageFile.LOAD_TRUNCATED_IMAGES = True

transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

dataset = 'office_home'
d_num_cls = {'office_home':65, 'PACS':7,'VLCS':5,'office31':31}
total_json = {}
for lr in ['1e-1','1e-2','5e-2','1e-3','5e-3']:
    total_json[lr] = 0
    for seed in [0,1,2]:
        domain_dict = {
            'office_home' : ['A','C','P','R'],
            'PACS' : ['A','C','P','S'],
            'VLCS' : ['C','L','S','V'],
            'office31' : ['A','D','W']
        }
        permute = [i+j for i in domain_dict[dataset] for j in domain_dict[dataset] if i!=j]

        netF = network.ResBase(res_name='resnet50').cuda()
        netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=256).cuda()
        netC = network.feat_classifier(type='wn', class_num=d_num_cls[dataset], bottleneck_dim=256).cuda()
        
        if seed == 0:
            fol_name = f'seed2020_lr{lr}'
        elif seed == 1:
            fol_name = f'seed2021_lr{lr}'
        elif seed == 2:
            fol_name = f'seed2022_lr{lr}'

        if lr == '1e-2':
            fol_name = fol_name[:-7]

        for i in permute:
            root_dir = f'/nas/home/tmddnjs3467/domain-generalization/SHOT/object/ckps/target_samesplit_{fol_name}/uda'
            
            modelpath = root_dir + f'/{dataset}/{i}/target_F_par_0.3.pt'   
            netF.load_state_dict(torch.load(modelpath))
            modelpath = root_dir + f'/{dataset}/{i}/target_B_par_0.3.pt'   
            netB.load_state_dict(torch.load(modelpath))
            modelpath = root_dir + f'/{dataset}/{i}/target_C_par_0.3.pt'    
            netC.load_state_dict(torch.load(modelpath))
            netC.eval()

            # data load
            data_path = os.path.join('/data/domainbed',dataset)
            domains = sorted([f.name for f in os.scandir(data_path) if f.is_dir()])
            path = f'/data/domainbed/{dataset}/{domains[domain_dict[dataset].index(i[-1])]}'
            target_dataset = ImageFolder(os.path.join(path))
            keys = list(range(len(target_dataset)))
            classes = target_dataset.targets
            
            tr_idx, te_idx = train_test_split(keys,test_size=0.2,random_state=seed,stratify=classes)
            _, test_dataset = _SplitDataset(target_dataset, tr_idx), _SplitDataset(target_dataset, te_idx)
            test_dataset.transform = transform
            test_loader = DataLoader(test_dataset,batch_size=32*3,shuffle=False,num_workers=8)
            
            _, snd, _, _ =test_and_nd(100, test_loader, f'./snd_results/{dataset}_snd_log', netF, netB, netC)
            total_json[lr] += snd

save_folder = './snd_results'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
with open(os.path.join(save_folder,f'{dataset}_snd.json'),'w') as f:
    json.dump(total_json, f)
