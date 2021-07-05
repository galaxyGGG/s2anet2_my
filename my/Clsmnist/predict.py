# Author: Arashi
# Time: 2021/6/29 上午9:23
# Desc:
import torch
from my.Clsmnist import model
import my.Clsmnist.loadDataMnist as loadData
import PIL.Image as Image
def modelLoad(weightName,classes):
    weightDict = torch.load(weightName)
    weight = weightDict['weight']
    modelName = weightDict['modelName']
    net = getattr(model, modelName)(len(classes))
    net.load_state_dict(state_dict=weight)
    return net

def imagePred(model,image,transform = loadData.valTransform):
    imageTensor = transform(image)
    pred = model.forward(imageTensor.unsqueeze(0))
    predSoftmax = torch.softmax(pred,dim=1)
    result = torch.argmax(predSoftmax,dim=1)
    return result.squeeze(0)

def pathPred(net,imagePath,classes):
    image = Image.open(imagePath)
    result = imagePred(net,image)
    result = result.squeeze().item()
    return classes[result]

if __name__ == '__main__':

    classes = loadData.dataSplit()['testset'].classes
    weightName = 'ResNetReDefine_SGD_warships_lr0.01_Seed10000_best.pkl'
    net = modelLoad(weightName,classes=classes)
    net.eval()

    imagePath = '/home/jyc/arashi/data/HRSC2016_cls/warships/test/医疗船/100001227_2.png'
    result = pathPred(net,imagePath,classes=classes)
    print()



