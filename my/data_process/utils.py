# Author: Arashi
# Time: 2021/6/25 下午5:07
# Desc: 数据处理中的各种工具
import os

import cv2
import numpy as np
from tqdm import tqdm


def check_points(points, image_id, w, h):
    """
    # 用于目标检测中,检查是否数据越界,以及是否有小目标
    :param points: 输入为顶点坐标
    :return:
    """
    flag_good = True
    new_cords = []
    for i in range(len(points)//2):
        cord0 = int(points[i * 2])
        cord1 = int(points[i * 2 + 1])
        # 判断数据越界
        if cord0 < 0:
            print("x < 0 ", image_id, str(cord0), str(cord1), "\n")
            cord0 = 0
            flag_good = False

            # break
        elif cord0 > w:
            print("x > width ", image_id, str(cord0), str(cord1), "\n")
            cord0 = w
            flag_good = False
            # break
        if cord1 < 0:
            print("y < 0 ", image_id, str(cord0), str(cord1), "\n")
            cord1 = 0
            flag_good = False
            # break
        elif cord1 > h:
            print("y > height ", image_id, str(cord0), str(cord1), "\n")
            cord1 = h
            flag_good = False
            # break
        points[i * 2] = cord0
        points[i * 2 + 1] = cord1
        new_cords.append([cord0,cord1])
    # 筛选掉了像素数小于10的小目标
    if cv2.contourArea(contour=np.float32(new_cords)) <= 10:
        print('小目标', image_id)
        flag_good = False
    return flag_good


def get_classes(labelTxt_dir,save_dir = ".."):
    """
    # 从dota格式的labelTxt文件夹中获取所有的类别
    :param labelTxt_dir: 标注文件夹路径
    :return:
    """
    dict_cls={}
    # 保存类别文件夹路径
    if save_dir == "..":
        save_dir = os.path.join(labelTxt_dir,save_dir)
    txt_files = os.listdir(labelTxt_dir)
    for txt in txt_files:
        txt_path = os.path.join(labelTxt_dir,txt)
        with open(txt_path,"r") as f:
            for line in f.readlines():
                line = line.strip().split()
                if len(line) != 10:
                    continue
                cls = line[-2]
                if not cls in dict_cls.keys():
                    dict_cls[cls] = 0
                dict_cls[cls] += 1
    print(dict_cls.items())
    print(dict_cls.keys())
    # 保存classes.txt
    with open(os.path.join(save_dir, "classes.txt"), "w") as f:
        # 保存classes_count.txt
        with open(os.path.join(save_dir, "classes_count.txt"), "w") as f2:
            for key, val in dict_cls.items():
                f.write(key + "\n")
                f2.write("%s: %s\n" % (key, val))
    print("num_classes:" + str(len(dict_cls.keys())))

def cal_mean_std(dir):
    img_filenames = os.listdir(dir)
    m_list, s_list = [], []
    for img_filename in tqdm(img_filenames):
        img = cv2.imread(os.path.join(dir,img_filename))
        # img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print(m[0][::-1])
    print(s[0][::-1])




if __name__ == '__main__':
    # txt_dir = "/home/jyc/arashi/data/HRSC2016/FullDataSet/labelTxt"
    # get_classes(txt_dir)

    # 计算方差
    cal_mean_std("/home/amax/ganlan/arashi/data/HRSC2016_dataset_800/trainval_split/images")