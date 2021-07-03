# Author: Arashi
# Time: 2021/7/2 下午2:49
# Desc:  测试集在做完检测后，会把大图的结果保存到merged文件夹txt中
#        把这个检测结果中需要进行小类识别的，再输入到分类网络中

import os
import shutil
from torchvision import datasets
from my.Clsmnist import predict
from my.data_process.data_det2cls import img_paste
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

def read_txts(merged_dir,img_dir,ext=".bmp"):
    # 新结果保存目录
    merged_dir_new = os.path.join(merged_dir,"../merged_cls")
    if not os.path.exists(merged_dir_new):
        os.mkdir(merged_dir_new)

    # *** 挨个打开文件，然后送进分类网络 ***
    txts = os.listdir(merged_dir)
    # 用于保存新的分类结果
    res_new_dict = {}
    for txt in txts:
        #  获取类名
        cls_name = txt.split(".txt")[0]
        txt_path = os.path.join(merged_dir, txt)
        # 根据类名，判断使用什么分类模型
        if cls_name == "aircraft_carrier":
            trainset = datasets.ImageFolder(
                root='/home/jyc/arashi/data/HRSC2016_cls/aircraft_carrier/images')
            classes = trainset.classes
            # classes = ["企业级航母","尼米兹级航母","俄罗斯库兹涅佐夫号航母","中途号航母"]
            weightName = '/home/jyc/arashi/PycharmProjects/ClsMnist/weight/ResNetReDefine_SGD_hangmu_lr0.01_Seed10000_trainAcc0.97_testAcc1.00.pkl'
        elif cls_name == "warship":
            # classes = ["阿利伯克级驱逐舰","圣安东尼奥级两栖船坞运输舰","塔拉瓦级通用两栖攻击舰","琵琶形军舰","提康德罗加级巡洋舰","佩里级护卫舰","尾部OX头部圆指挥舰",
            #  "奥斯汀级两栖船坞运输舰","惠德贝岛级船坞登陆舰","医疗船"]
            trainset = datasets.ImageFolder(
                root='/home/jyc/arashi/data/HRSC2016_cls/warships/train')
            classes = trainset.classes
            print(classes)
            weightName = '/home/jyc/arashi/PycharmProjects/ClsMnist/weight/ResNetReDefine_SGD_warships_newcat_lr0.01_Seed10000_trainAcc0.99_testAcc0.93.pkl'
        else:
            # 直接保存
            shutil.copy(txt_path,os.path.join(merged_dir_new,txt))
            continue

        # *** 进行分类 ***
        #  创建分类模型
        net = predict.modelLoad(weightName, classes=classes)
        net.eval()
        with open(txt_path,"r") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                res = line.strip().split()
                img_path = os.path.join(img_dir,res[0]+ext)

                # 根据bbox坐标，裁剪图片中的目标
                bbox = [int(float(loc)) for loc in res[2:10]]
                polygon = [[bbox[0], bbox[1]], [bbox[2], bbox[3]], [bbox[4], bbox[5]], [bbox[6], bbox[7]]]
                # 根据bbox裁剪图片
                img_crop = img_paste(img_path, polygon,
                                     "/my/Clsmnist/temp/temp.jpg",
                                     crop_size=512)

                # 结果送入分类网络，识别出小类，保存到临时变量
                result = predict.imagePred(net, img_crop)
                result = result.squeeze().item()
                cls_res = classes[result]

                # *** 保存到结果字典中 ***
                if not cls_res in res_new_dict.keys():
                    res_new_dict[cls_res]=[]
                res_new_dict[cls_res].append(line)

    # *** 做完分类后，保存结果 ***
    # 保存到pkl文件
    with open(os.path.join(merged_dir_new,"res_after_cls.pkl"),"wb") as f_pkl:
        pickle.dump(res_new_dict,f_pkl)
    # 保存到txt中
    for key in res_new_dict.keys():
        with open(os.path.join(merged_dir_new,key+".txt"),"w") as f:
            for line in res_new_dict[key]:
                f.write(line)


if __name__ == '__main__':
    merged_dir = "/home/jyc/arashi/PycharmProjects/s2anet2_my/work_dirs/s2anet_r50_fpn_1x_dota_hrsc2016_category/results_after_nms"
    img_dir = "/home/jyc/arashi/data/HRSC2016/FullDataSet/AllImages"

    read_txts(merged_dir,img_dir,".bmp")



