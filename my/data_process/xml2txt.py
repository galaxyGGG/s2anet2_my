# 转化FAIR1M数据集为dota的txt格式
import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
from tqdm import tqdm

# 小类合并为大类
dict_merge_cls = {
    "airplane": ['other-airplane', 'Boeing737', 'A220', 'A321', 'Boeing747', 'ARJ21', 'A330', 'A350', 'C919',
                 'Boeing777', 'Boeing787'],
    "road": ['Bridge', 'Intersection', 'Roundabout'],
    "car": ['SmallCar', 'Van', 'DumpTruck', 'CargoTruck', 'TruckTractor', 'Trailer', 'other-vehicle', 'Bus', 'Tractor',
            'Excavator'],
    "ship": ['Motorboat', 'DryCargoShip', 'FishingBoat', 'LiquidCargoShip', 'PassengerShip', 'EngineeringShip',
             'other-ship', 'Warship', 'Tugboat'],
    "court":['TennisCourt','FootballField','BaseballField', 'BasketballCourt']
}

dict_cls = {}

def convert_annotation(image_id,xml_dir,txt_dir,merge_cls = False):
    """
    # 转换这一张图片的坐标表示方式（格式）,即读取xml文件的内容，存放在txt文件中。
    :param image_id:
    :return:
    """
    in_file = open(os.path.join(xml_dir,'%s.xml'%(image_id)))
    out_file = open(os.path.join(txt_dir,'%s.txt'%(image_id)), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    # 来源
    origin = root.find("source").find("origin").text
    out_file.write("imagesource:"+origin+"\n")
    out_file.write("gsd:\n")
    for obj in root.iter('object'):
        difficult = "0"
        cls = obj.find('possibleresult').find("name").text.replace(" ","")
        # 小类合并为大类
        if merge_cls:
            for key,value in dict_merge_cls.items():
                if cls in value:
                    cls=key
                    break
        if not cls in dict_cls.keys():
            dict_cls[cls] = 0
        dict_cls[cls] += 1
        points = obj.find('points')[:-1]
        new_cords=[]
        for point in points:
            cords = point.text.split(",")
            cord0 = int(round(float(cords[0])))
            cord1 = int(round(float(cords[1])))
            # 判断数据越界
            if cord0<0:
                print("x < 0 ",image_id,str(cord0),str(cord1),"\n")
                cord0 = 0
                break
            elif cord0 > w:
                print("x > width ", image_id, str(cord0), str(cord1), "\n")
                cord0 = w
                break
            if cord1 < 0:
                print("y < 0 ", image_id, str(cord0), str(cord1), "\n")
                cord1 = 0
                break
            elif cord1 > h:
                print("y > height ", image_id, str(cord0), str(cord1), "\n")
                cord1 = h
                break
            new_cords.append([cord0, cord1])
        else:
            # continue
            if cv2.contourArea(contour=np.float32(new_cords)) <= 10:  # 筛选掉了像素数小于10的小目标
                print('小目标', image_id)
                continue
            for cord in new_cords:
                out_file.write(str(cord[0])+" "+str(cord[1])+" ")
            out_file.write(cls + " " + difficult + "\n")

if __name__ == '__main__':
    xml_dir = "/home/amax/ganlan/arashi/data/FAIR1M/labelXmls"
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
        convert_annotation(img_id,xml_dir,txt_dir,merge_cls = True)

    print(dict_cls)
    print(dict_cls.keys())
    print("num_classes:"+str(len(dict_cls.keys())))
