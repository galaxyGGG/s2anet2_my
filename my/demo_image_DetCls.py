import os
import random

import cv2

from demo.demo_inference import show_result_rbox
from mmdet.apis import init_detector, inference_detector
import matplotlib.pyplot as plt
from my.Clsmnist import predict
from my.data_process.data_det2cls import img_paste
from torchvision import datasets
from my.DOTA import DOTA
config_file = '/home/jyc/arashi/PycharmProjects/s2anet2_my/my/config/s2anet_r50_fpn_1x_dota_hrsc2016_category.py'
checkpoint_file = '/home/jyc/arashi/PycharmProjects/s2anet2_my/checkpoints/work_dirs/s2anet_r50_fpn_1x_dota_hrsc2016_categoryNew/epoch_74.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
# img_path = '/home/jyc/arashi/data/HRSC2016_dataset_1024/test_split/images/100000238__1__397___0.bmp'  # or img = mmcv.imread(img), which will only load it once
img_dir = "/home/jyc/arashi/data/HRSC2016_dataset_newcat_1024/test_split/images"
examplesplit = DOTA("/home/jyc/arashi/data/HRSC2016_dataset_newcat_1024/test_split")
# imgids = examplesplit.getImgIds()
count = 0
for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir,img_name)
    result = inference_detector(model, img_path)

    img,bboxes = show_result_rbox(img_path,
                         result,
                         class_names=model.CLASSES,
                         scale=1.0,
                         threshold=0.5,
                         colormap=None,
                         show_label=True)
    is_hit =False
    plt.figure()
    # 裁剪小图进行细分类
    anns = examplesplit.loadAnns(imgId=os.path.splitext(img_name)[0])
    plt.subplot(1, 2, 1)

    examplesplit.showAnns(anns, os.path.splitext(img_name)[0], 1)
    plt.subplot(1, 2, 2)
    # plt.figure()
    for bbox in bboxes:
        bbox=bbox.strip().split(" ")
        print(bbox)
        line = [int(float(loc)) for loc in bbox[2:10]]
        polygon = [[line[0], line[1]], [line[2], line[3]], [line[4], line[5]], [line[6], line[7]]]
        # 根据bbox裁剪图片
        img_crop = img_paste(img_path, polygon, "/home/jyc/arashi/PycharmProjects/s2anet2_my/my/Clsmnist/temp/temp.jpg", crop_size=512)


        weightName = ""
        classes = []
        # 判断使用什么分类模型
        if bbox[-1] == "aircraft_carrier":
            trainset = datasets.ImageFolder(
                root='/home/jyc/arashi/data/HRSC2016_cls/aircraft_carrier/images')
            classes = trainset.classes
            # classes = ["企业级航母","尼米兹级航母","俄罗斯库兹涅佐夫号航母","中途号航母"]
            weightName = '/home/jyc/arashi/PycharmProjects/ClsMnist/weight/ResNetReDefine_SGD_lr0.01_Seed10000.pkl'
        elif bbox[-1] == "warship":
            # classes = ["阿利伯克级驱逐舰","圣安东尼奥级两栖船坞运输舰","塔拉瓦级通用两栖攻击舰","琵琶形军舰","提康德罗加级巡洋舰","佩里级护卫舰","尾部OX头部圆指挥舰",
            #  "奥斯汀级两栖船坞运输舰","惠德贝岛级船坞登陆舰","医疗船"]
            trainset = datasets.ImageFolder(
                root='/home/jyc/arashi/data/HRSC2016_cls/warships/train')
            classes = trainset.classes
            print(classes)
            weightName = '/home/jyc/arashi/PycharmProjects/ClsMnist/weight/ResNetReDefine_SGD_warships_newcat_lr0.01_Seed10000_trainAcc0.99_testAcc0.93.pkl'
        if len(classes) == 0 or weightName =="":
            continue
        net = predict.modelLoad(weightName, classes=classes)
        net.eval()
        result = predict.imagePred(net, img_crop)
        result = result.squeeze().item()
        cls_res = classes[result]
        print(cls_res)

        # *** 画图模块 ***
        # 画出原图
        # raw_img_name = img_name.split("_")[0] + "."+img_name.split(".")[-1]
        for i in range(3):
            cv2.line(img, (line[i * 2], line[i * 2 + 1]), (line[(i + 1) * 2], line[(i + 1) * 2 + 1]), color=(255,0,0),
                     thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img, (line[6], line[7]), (line[0], line[1]), color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)
        # cv2.putText(img, cls_res, (line[0], line[1] + 10),
        #             color=(255,0,0), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)

        # 按类名加载图片
        # imgids = examplesplit.getImgIds(catNms=['plane'])
        # img = examplesplit.loadImgs(imgids)
        # plt.figure()
        # plt.title(img_name)
        # plt.show()


        # plt.text((line[0]+line[4])//2, (line[1]+line[5])//2 + 10,cls_res,color="red",size=10)
        plt.text(line[0], line[1] + random.randint(0,10), "%s %.2f"%(cls_res,float(bbox[1])), color="red", size=10)
        is_hit = True

    if not is_hit == True:
        plt.close()
        continue

    plt.subplot(1,2,2)
    plt.imshow(img)
    # plt.show()
    plt.savefig(os.path.join(img_dir,"../det_cls_res",img_name.split(".")[0]+".jpg"))
    count +=1
    # if count==20:
    #     break

# or save the visualization results to image files
# show_result(img, result, model.CLASSES, out_file='result.jpg')