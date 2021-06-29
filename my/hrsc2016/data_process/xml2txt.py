# Desc: 转化FAIR1M数据集为dota的txt格式

import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
from tqdm import tqdm

from mmdet.core import rotated_box_to_poly_single, poly_to_rotated_box_single
from my.data_process.utils import check_points

dict_cls = {}
is_save = True

def convert_annotation(image_id,xml_dir,txt_dir,classes_dict, wanted_classes = []):
    """
    # 转换这一张图片的坐标表示方式（格式）,即读取xml文件的内容，存放在txt文件中。
    :param image_id:
    :return:
    """
    in_file = open(os.path.join(xml_dir,'%s.xml'%(image_id)))
    out_file = open(os.path.join(txt_dir,'%s.txt'%(image_id)), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    w = int(root.find('Img_SizeWidth').text)
    h = int(root.find('Img_SizeHeight').text)
    # 来源
    # origin = root.find("source").find("origin").text
    out_file.write("imagesource: \n")
    out_file.write("gsd:\n")

    for obj in root.find("HRSC_Objects").iter('HRSC_Object'):
        difficult = obj.find("difficult").text
        # 获取class_id
        cls_id = obj.find('Class_ID').text
        # 对应的class名称
        cls = classes_dict[cls_id]
        # cls = cls_id  # 直接等于id

        if not len(wanted_classes) == 0:
            if cls not in wanted_classes:
                continue

        if not cls in dict_cls.keys():
            dict_cls[cls] = 0
        dict_cls[cls] += 1

        # rbox 转 poly表示
        rbox=[obj.find("mbox_cx").text, obj.find("mbox_cy").text, obj.find("mbox_w").text, obj.find("mbox_h").text, obj.find("mbox_ang").text]
        rbox = [float(x) for x in rbox]
        points = rotated_box_to_poly_single(rbox)

        # 判断数据是否越界
        is_good = check_points(points,image_id,w,h)
        # 这里决定一下是不是要保留越界的点
        if not is_good:
            if not is_save:
                continue
        for cord in points:
            out_file.write(str(int(round(cord))) + " ")
        out_file.write(cls + " " + difficult + "\n")



def get_classes_from_xml(xml_file):
    """
    # 从数据集配置文件中获取所有类别信息
    :param xml_file:
    :return:
    """
    classes_dict = {}
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for cls in root.find("HRSC_Classes").iter("HRSC_Class"):
        cls_id = cls.find("Class_ID").text
        if cls_id not in classes_dict.keys():
            classes_dict[cls_id] = cls.find('Class_Name').text
    return classes_dict


if __name__ == '__main__':
    xml_dir = "/home/jyc/arashi/data/HRSC2016/FullDataSet/Annotations"
    sysdata = "/home/jyc/arashi/data/HRSC2016/FullDataSet/sysdata.xml"
    # 筛选部分类别
    wanted_classes = ['阿利伯克级驱逐舰', '圣安东尼奥级两栖船坞运输舰', '塔拉瓦级通用两栖攻击舰',
                      '琵琶形军舰', '企业级航母', '提康德罗加级巡洋舰', '佩里级护卫舰', '尾部OX头部圆指挥舰', '奥斯汀级两栖船坞运输舰', '潜艇', '惠德贝岛级船坞登陆舰',
                      '尼米兹级航母', '军舰', '俄罗斯库兹涅佐夫号航母', '医疗船', '中途号航母', '蓝岭级指挥舰', '小鹰级航母',
                      '航母']
    classes_dict = get_classes_from_xml(sysdata)

    txt_dir = ""
    if txt_dir == "":
        txt_dir = os.path.join(xml_dir,"../labelTxt")
    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)

    files = os.listdir(xml_dir)
    for file in tqdm(files):
        if not os.path.splitext(file)[-1]==".xml":
            continue
        img_id = os.path.splitext(file)[:-1]
        convert_annotation(img_id,xml_dir,txt_dir,classes_dict, wanted_classes) #如果不需要筛选类别，则删除wanted_classes

    print(dict_cls)
    print(dict_cls.keys())
    # 保存classes.txt
    with open(os.path.join(xml_dir,"../classes.txt"),"w") as f:
        # 保存classes_count.txt
        with open(os.path.join(xml_dir, "../classes_count.txt"), "w") as f2:
            for key,val in dict_cls.items():
                f.write(key+"\n")
                f2.write("%s: %s\n"%(key,val))
            f2.write(str(dict_cls.keys()))
    print("num_classes:"+str(len(dict_cls.keys())))
