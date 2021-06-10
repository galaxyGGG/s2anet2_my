from torchvision import models
import torch
net = models.densenet121(pretrained=False)
print(net)
print(net(torch.zeros(1,3,256,256)).shape)