# 把数据集中的各个目标提取出来，按照一定大小在原图中裁剪目标，不够的用黑色填充
# 生成用于分类的小图片，按照分类保存到不同文件夹下
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

classes = ['other-airplane', 'Bridge', 'SmallCar', 'Van', 'DumpTruck', 'CargoTruck', 'Motorboat', 'Boeing737',
           'TruckTractor', 'Intersection', 'A220', 'A321', 'TennisCourt', 'FootballField', 'DryCargoShip',
           'FishingBoat', 'Trailer', 'other-vehicle', 'LiquidCargoShip', 'PassengerShip', 'EngineeringShip',
           'Excavator', 'BaseballField', 'other-ship', 'BasketballCourt', 'Bus', 'Boeing747', 'Tractor', 'Warship',
           'Tugboat', 'ARJ21', 'A330', 'A350', 'C919', 'Boeing777', 'Boeing787', 'Roundabout']
# 飞机 船 20类
classes = ['Motorboat', 'Boeing737', 'A220', 'A321', 'DryCargoShip',
           'FishingBoat', 'LiquidCargoShip', 'PassengerShip', 'EngineeringShip',
           'Boeing747', 'Warship',
           'Tugboat', 'ARJ21', 'A330', 'A350', 'C919', 'Boeing777', 'Boeing787']

def img_paste(img_path, polygon, save_file_path,crop_size=512):
    """
    裁剪多边形
    :param img_path:
    :param polygon:
    :param save_file_path:
    :param crop_size:  最后的裁剪尺寸
    :return:
    """


    image = cv2.imread(img_path)
    im = np.zeros(image.shape[:2], dtype="uint8")
    polygon = np.array(polygon)
    # 外接矩形坐标
    right_up = np.max(polygon,axis=0)
    left_down = np.min(polygon, axis=0)

    polygon = polygon.reshape((-1, 1, 2))
    # 把所有的点画出来
    cv2.polylines(im, [polygon], 1, 255)
    # 把所有点连接起来，形成封闭区域
    cv2.fillPoly(im, [polygon], 255)
    # plt.imshow(im)
    # plt.show()

    # print(np.maximum(mask,))
    # 将连接起来的区域对应的数组和原图对应位置按位相与
    masked = cv2.bitwise_and(image, image, mask=im)
    # plt.imshow(masked)
    # plt.show()
    # cv2中的图片是按照bgr顺序生成的，我们需要按照rgb格式生成
    b, g, r = cv2.split(masked)
    masked = cv2.merge([r, g, b])
    # plt.imshow(masked)
    # plt.show()

    ### 裁剪图片，先把目标的外接矩形裁出来，再扩展到统一尺寸crop_size
    ### 小的图片直接用0填充，大的图片缩放到该尺寸
    # 目标外接正矩形
    crop_masked = masked[left_down[1]:right_up[1], left_down[0]:right_up[0], :]
    # plt.figure()
    # plt.imshow(crop_masked)
    # plt.show()

    # 判断图片是否大于crop_size
    hw = np.array(crop_masked.shape[:2])
    if max(hw) > crop_size:
        # 缩放
        zoom_ratio = crop_size/max(hw)
        hw = hw * zoom_ratio
        hw[np.argmax(hw)] = crop_size

        crop_masked = cv2.resize(crop_masked,(int(hw[1]),int(hw[0])),0,0)
        # 不够的地方再填充


        # plt.figure()
        # plt.imshow(crop_masked)
        # plt.show()
    # 填充图片到（crop_size,cropsize）
    expand_top = int((crop_size-crop_masked.shape[0])/2)
    expand_bottom = crop_size - crop_masked.shape[0] - expand_top
    expand_left = int((crop_size-crop_masked.shape[1])/2)
    expand_right = crop_size - crop_masked.shape[1] - expand_left
    crop_masked = cv2.copyMakeBorder(crop_masked, expand_top, expand_bottom, expand_left, expand_right, cv2.BORDER_CONSTANT,
                             value=(0, 0, 0))
    assert crop_masked.shape[:2] == (crop_size, crop_size)

    # plt.imshow(crop_masked)
    # plt.show()

    cv2.imwrite(save_file_path, crop_masked)


def extract_object(txt_dir,img_dir,ext=".png"):
    # 小图保存路径
    save_path = os.path.join(txt_dir, "..", "extracted_obj")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for cls in classes:
        cls_dir = os.path.join(save_path,cls)
        if not os.path.exists(cls_dir):
            os.mkdir(cls_dir)
    # 标注文件保存路径
    ann_path = os.path.join(save_path,"..","label_cls.txt")
    with open(ann_path,"w") as f_ann:
        for txt_file in tqdm(os.listdir(txt_dir)):
            img_name = os.path.splitext(txt_file)[0] + ext
            img_path = os.path.join(img_dir, img_name)
            txt_file=os.path.join(txt_dir,txt_file)
            with open(txt_file,"r") as f:
                for ind_img,line in enumerate(f.readlines()):
                    # print(ind_img,line)
                    line=line.strip().split(" ")
                    if len(line) != 10:
                        continue
                    save_file_name = os.path.splitext(img_name)[0] + "_" + str(ind_img) + '.png'


                    # 标注部分
                    cls = line[-2]
                    if cls not in classes:
                        continue
                    ind_cls = classes.index(cls)
                    f_ann.write(save_file_name + " " + str(ind_cls) + "\n")

                    # 裁剪图片
                    save_file_path = os.path.join(save_path,cls, save_file_name)
                    for i in range(8):
                        line[i] = int(line[i])
                    polygon=[[line[0],line[1]],[line[2],line[3]],[line[4],line[5]],[line[6],line[7]]]

                    img_paste(img_path, polygon, save_file_path)




if __name__ == '__main__':
    txt_dir = "/home/jyc/arashi/data/FAIR1M_dataset/test/labelTxt"
    img_dir = "/home/jyc/arashi/data/FAIR1M_dataset/test/images"
    extract_object(txt_dir,img_dir)