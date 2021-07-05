import torch
from torchvision import datasets,transforms
from torch.utils.data import Dataset
import sys
import numpy as np

#加上transforms
normalize=transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
Transform=transforms.Compose([
    transforms.RandomRotation(90),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    normalize
])
valTransform = transforms.Compose([
    transforms.ToTensor(), #将图片转换为Tensor,归一化至[0,1]
    normalize
])
class MNISTDataset(Dataset):
    def __init__(self,trainData) :
        super(MNISTDataset,self).__init__()
        self.trainData = trainData

    def __getitem__(self, index: int):
        data = self.trainData[index]
        image,label = data 
        image = torch.FloatTensor([np.array(image.convert('L'))])/255
        label = torch.ByteTensor(np.array(label).reshape(-1)).squeeze(-1).long()
        return image,label
    
    def __len__(self) -> int:
        return len(self.trainData)
    
def dataSplit():
    # trainData = datasets.MNIST(root=sys.path[0],train=True,download=True)
    # testData = datasets.MNIST(root=sys.path[0],train=False)
    trainset = datasets.ImageFolder(
    root='/home/jyc/arashi/data/HRSC2016_cls/warships/train',transform=Transform)
    testset = datasets.ImageFolder(
    root='/home/jyc/arashi/data/HRSC2016_cls/warships/test',transform=valTransform)
    # trainset = MNISTDataset(trainData)
    # testset = MNISTDataset(testData)
    return {'trainset':trainset,'testset':testset}

if __name__ == '__main__':
    dataset = dataSplit()
    trainset = dataset['trainset']
    print(len(dataset['trainset']))
    print(trainset[0][1])
    datasetImageNet = datasets.ImageFolder(root='/home/jyc/arashi/data/HRSC2016_cls/warships/train')
    print('class:',trainset.classes)
    print(len(datasetImageNet))
    print(len(datasetImageNet[1]))