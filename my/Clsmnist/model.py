from torch import nn
import torch 
from  torchvision import models
class FCNet(nn.Module):
    def __init__(self,inputShape=[1,28,28],outputShape=10):
        super(FCNet,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(inputShape[0]*inputShape[1]*inputShape[2],1024),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(1024,1024),nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(1024,10),nn.ReLU())
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class ResNetReDefine(nn.Module):
    def __init__(self,outputShape = 19):
        super(ResNetReDefine,self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.layer = nn.Linear(in_features=self.model.fc.in_features,out_features=outputShape)
        self.model.fc = self.layer
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    net = FCNet()
    print(models.resnet50())
    print(net)
