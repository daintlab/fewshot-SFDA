import torch.nn as nn
import torchvision

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
        return self.classifier(self.network(x))