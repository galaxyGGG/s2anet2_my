# 检查不合格数据(在fair1m数据集中容易报错)：越界及小于10像素
from tqdm import tqdm

from DOTA_devkit.DOTA import DOTA
import matplotlib.pyplot as plt
import cv2
import numpy as np

examplesplit = DOTA("/home/jyc/arashi/data/FAIR1M_dataset_1024/trainval_split")

imgids = examplesplit.getImgIds()
count_bad=0
for imgid in tqdm(imgids):
    anns = examplesplit.loadAnns(imgId=imgid)
    for ann in anns:
        cords = ann["poly"]
        for cord in cords:
            # 数据越界
            if cord[0] * cord[1]<0:
                print("\n负数", imgid,cords)
                count_bad+=1
                # examplesplit.showAnns(anns, imgid, 2)
                # plt.show()
        # 不足10像素
        if cv2.contourArea(contour=np.float32(cords)) <= 10:  # 筛选掉了像素数小于10的小目标
            print('\n小目标', imgid,cords)
            # examplesplit.showAnns(anns, imgid, 2)
            # plt.show()
            count_bad += 1
print(count_bad)