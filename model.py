import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
from collections import OrderedDict
import torch.nn.utils.weight_norm as weightNorm
from torchvision import models

BLOCKNAMES = {
        "stem": ["conv1", "bn1", "relu", "maxpool"],
        "block1": ["layer1"],
        "block2": ["layer2"],
        "block3": ["layer3"],
        "block4": ["layer4"],
    }

class ERM(nn.Module):
    def __init__(self,num_classes,arch='resnet50'):
        super(ERM,self).__init__()
        if arch == 'resnet50':
            self.network = torchvision.models.resnet50(pretrained=True)
        else:
            raise ValueError("Not implemented")
        n_features = self.network.fc.in_features
        
        del self.network.fc
        self.network.fc = nn.Identity()
        
        self.classifier = nn.Linear(n_features,num_classes)

    def forward(self,x):
        logit = self.classifier(self.network(x))
        return logit
    

class ERMWithFeature(nn.Module):
    def __init__(self,num_classes,feature_block=None,arch='resnet50'):
        super(ERMWithFeature,self).__init__()
        if arch == 'resnet50':
            self.network = torchvision.models.resnet50(pretrained=True)
        else:
            raise ValueError("Not implemented")
        n_features = self.network.fc.in_features
        
        del self.network.fc
        self.network.fc = nn.Identity()
        
        self.classifier = nn.Linear(n_features,num_classes)

        self.feature = []
        self.feature_layers = self.feature_hook(feature_block,BLOCKNAMES)
        
    def hook(self,module,input,output):
        self.feature.append(torch.mean(output,dim=[2,3]))
    
    def feature_hook(self,feature_block,block_names):
        if feature_block is None:
            feature_block = list(block_names.keys())
        else:
            feature_block = feature_block.split(',')
        feat_layers = []
        for block in feature_block:
            module_list = block_names.get(block,-1)
            if module_list == -1:
                raise ValueError("no such block name")
            feat_layers.append(module_list[-1])
        
        for n,m in self.network.named_modules():
            if n in feat_layers:
                m.register_forward_hook(self.hook)
        
        return feat_layers
                
    def forward(self,x,ret_feats=False):
        self.feature.clear()
        logit = self.classifier(self.network(x))
        if ret_feats:
            return logit,self.feature
        else:
            return logit

class SHOT(nn.Module):
    def __init__(self,ckpts,dataset='office_home',subset=False):
        super(SHOT,self).__init__()
        if subset:
            config = {
                'office_home' : {
                    'arch' : 'resnet50',
                    'class_num' : 25,
                    'bottleneck_dim' : 256
                },
                'office31' : {
                    'arch' : 'resnet50',
                    'class_num' : 10,
                    'bottleneck_dim' : 256
                },
                'VISDA-C' : {
                    'arch' : 'resnet101',
                    'class_num' : 6,
                    'bottleneck_dim' : 256
                }
            }
        else:
            config = {
                'office_home' : {
                    'arch' : 'resnet50',
                    'class_num' : 65,
                    'bottleneck_dim' : 256
                },
                'PACS' : {
                    'arch' : 'resnet50',
                    'class_num' : 7,
                    'bottleneck_dim' : 256
                },
                'VLCS' : {
                    'arch' : 'resnet50',
                    'class_num' : 5,
                    'bottleneck_dim' : 256
                },
                'office31' : {
                    'arch' : 'resnet50',
                    'class_num' : 31,
                    'bottleneck_dim' : 256
                },
                'terra_incognita' : {
                    'arch' : 'resnet50',
                    'class_num' : 8,
                    'bottleneck_dim' : 256
                },
                'VISDA-C' : {
                    'arch' : 'resnet101',
                    'class_num' : 12,
                    'bottleneck_dim' : 256
                }
            }
        self.netF = ResBase(res_name=config[dataset]['arch'])
        self.netB = feat_bottleneck(
            type='bn',
            feature_dim=self.netF.in_features,
            bottleneck_dim=config[dataset]['bottleneck_dim']
        )
        self.netC = feat_classifier(
            type='wn',
            class_num=config[dataset]['class_num'],
            bottleneck_dim=config[dataset]['bottleneck_dim']
        )
        self.netF.load_state_dict(torch.load(ckpts['netF']))
        self.netB.load_state_dict(torch.load(ckpts['netB']))
        self.netC.load_state_dict(torch.load(ckpts['netC']))
        # self.head = torch.nn.Linear(256, 4)
    # def forward(self,x):
    def forward(self,x,flag=True):
        if flag:
            return self.netC(self.netB(self.netF(x)))
        else:
            feat = self.netB(self.netF(x))
            output = self.netC(feat)
            c_output = self.head(feat)
            return output, c_output
    def infer(self,x):
        return self.netC(self.netB(self.netF(x)))
    def get_feature(self,x):
        return self.netF(x)
    def get_output(self,x):
        return self.netC(self.netB(x))
class SHOT_fe(nn.Module):
    def __init__(self,ckpts,dataset='office_home',subset=False):
        super(SHOT_fe,self).__init__()
        if subset:
            config = {
                'office_home' : {
                    'arch' : 'resnet50',
                    'class_num' : 25,
                    'bottleneck_dim' : 256
                },
                'PACS' : {
                    'arch' : 'resnet50',
                    'class_num' : 7,
                    'bottleneck_dim' : 256
                },
                'VLCS' : {
                    'arch' : 'resnet50',
                    'class_num' : 5,
                    'bottleneck_dim' : 256
                },
                'office31' : {
                    'arch' : 'resnet50',
                    'class_num' : 10,
                    'bottleneck_dim' : 256
                },
                'terra_incognita' : {
                    'arch' : 'resnet50',
                    'class_num' : 8,
                    'bottleneck_dim' : 256
                },
                'VISDA-C' : {
                    'arch' : 'resnet101',
                    'class_num' : 6,
                    'bottleneck_dim' : 256
                }
            }
        else:
            config = {
                'office_home' : {
                    'arch' : 'resnet50',
                    'class_num' : 65,
                    'bottleneck_dim' : 256
                },
                'PACS' : {
                    'arch' : 'resnet50',
                    'class_num' : 7,
                    'bottleneck_dim' : 256
                },
                'VLCS' : {
                    'arch' : 'resnet50',
                    'class_num' : 5,
                    'bottleneck_dim' : 256
                },
                'office31' : {
                    'arch' : 'resnet50',
                    'class_num' : 31,
                    'bottleneck_dim' : 256
                },
                'terra_incognita' : {
                    'arch' : 'resnet50',
                    'class_num' : 8,
                    'bottleneck_dim' : 256
                },
                'VISDA-C' : {
                    'arch' : 'resnet101',
                    'class_num' : 12,
                    'bottleneck_dim' : 256
                }
            }
        self.netF = ResBase(res_name=config[dataset]['arch'])
        self.netB = feat_bottleneck(
            type='bn',
            feature_dim=self.netF.in_features,
            bottleneck_dim=config[dataset]['bottleneck_dim']
        )
        self.netC = feat_classifier(
            type='wn',
            class_num=config[dataset]['class_num'],
            bottleneck_dim=config[dataset]['bottleneck_dim']
        )
        if ckpts is not None:
            state_dict = torch.load(ckpts)
            F = OrderedDict()
            B = OrderedDict()
            C = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('netF'):
                    F[k.replace('netF.', '')] = v
                elif k.startswith('netB'):
                    B[k.replace('netB.', '')] = v
                elif k.startswith('netC'):
                    C[k.replace('netC.', '')] = v
            self.netF.load_state_dict(F)
            self.netB.load_state_dict(B)
            self.netC.load_state_dict(C)
        
    def forward(self,x):
        return self.netC(self.netB(self.netF(x)))

class SHOT_ODA(nn.Module):
    def __init__(self,ckpts,dataset='office_home'):
        super(SHOT_ODA,self).__init__()
        config = {
            'office_home' : {
                'arch' : 'resnet50',
                'class_num' : 25,
                'bottleneck_dim' : 256
            },
            'PACS' : {
                'arch' : 'resnet50',
                'class_num' : 7,
                'bottleneck_dim' : 256
            },
            'VLCS' : {
                'arch' : 'resnet50',
                'class_num' : 5,
                'bottleneck_dim' : 256
            },
            'office31' : {
                'arch' : 'resnet50',
                'class_num' : 31,
                'bottleneck_dim' : 256
            },
            'VISDA-C' : {
                'arch' : 'resnet101',
                'class_num' : 12,
                'bottleneck_dim' : 256
            }
        }
        self.netF = ResBase(res_name=config[dataset]['arch'])
        self.netB = feat_bottleneck(
            type='bn',
            feature_dim=self.netF.in_features,
            bottleneck_dim=config[dataset]['bottleneck_dim']
        )
        self.netC = feat_classifier(
            type='wn',
            class_num=config[dataset]['class_num'],
            bottleneck_dim=config[dataset]['bottleneck_dim']
        )
        self.netF.load_state_dict(torch.load(ckpts['netF']))
        self.netB.load_state_dict(torch.load(ckpts['netB']))
        self.netC.load_state_dict(torch.load(ckpts['netC']))
    
    def forward(self,x):
        return self.netC(self.netB(self.netF(x)))

class IMGNET(nn.Module):
    def __init__(self,ckpts=None,dataset='office_home'):
        super(IMGNET,self).__init__()
        config = {
            'office_home' : {
                'arch' : 'resnet50',
                'class_num' : 65,
                'bottleneck_dim' : 256
            },
            'PACS' : {
                'arch' : 'resnet50',
                'class_num' : 7,
                'bottleneck_dim' : 256
            },
            'VLCS' : {
                'arch' : 'resnet50',
                'class_num' : 5,
                'bottleneck_dim' : 256
            },
            'office31' : {
                'arch' : 'resnet50',
                'class_num' : 31,
                'bottleneck_dim' : 256
            },
            'terra_incognita' : {
                'arch' : 'resnet50',
                'class_num' : 8,
                'bottleneck_dim' : 256
            },
            'VISDA-C' : {
                'arch' : 'resnet101',
                'class_num' : 12,
                'bottleneck_dim' : 256
            }
        }
        self.netF = ResBase(res_name=config[dataset]['arch'])
        self.netC = torch.nn.Linear(self.netF.in_features, config[dataset]['class_num'])
        if ckpts is not None:
            state_dict = torch.load(ckpts)
            F = OrderedDict()
            C = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('netF'):
                    F[k.replace('netF.', '')] = v
                elif k.startswith('netC'):
                    C[k.replace('netC.', '')] = v
            self.netF.load_state_dict(F)
            self.netC.load_state_dict(C)
        self.linear_transform = torch.nn.Linear(self.netF.in_features, 256)
    def forward(self,x):
        return self.netC(self.netF(x))
    def infer(self,x):
        return self.netC(self.netF(x))
    def get_feature(self,x):
        return self.netF(x)
    

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, 
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}
        
class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x
