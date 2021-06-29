# Desc: 把数据集中的各个目标提取出来，按照一定大小在原图中裁剪目标，不够的用黑色填充
#       生成用于分类的小图片，按照分类保存到不同文件夹下
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from mmdet.core import poly_to_rotated_box_single
from my.image.utils_img import rotate_bound,rotateImage


def get_cls_list(cls_txt):
    """
    # 从txt中读取classes保存到list中
    :param cls_txt:
    :return:
    """
    cls_dict = []
    with open(cls_txt,"r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            cls_dict.append(line)
    return cls_dict



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
    # im = np.zeros(image.shape[:2], dtype="uint8")
    polygon = np.array(polygon)

    rbox2 = poly_to_rotated_box_single(list(polygon.reshape(-1)))  # 转为rbox，查看旋转角度
    angle = np.rad2deg(float(rbox2[-1]))
    # polygon = polygon.reshape(-1)
    img_crop = rotateImage(image,angle,tuple(polygon[0]),tuple(polygon[1]),tuple(polygon[2]),tuple(polygon[3]))
    # plt.imshow(img_crop)
    # plt.show()

    # # 外接矩形坐标
    # right_up = np.max(polygon,axis=0)
    # left_down = np.min(polygon, axis=0)
    #
    # polygon = polygon.reshape((-1, 1, 2))
    # # 把所有的点画出来
    # cv2.polylines(im, [polygon], 1, 255)
    # # 把所有点连接起来，形成封闭区域
    # cv2.fillPoly(im, [polygon], 255)
    # # plt.imshow(im)
    # # plt.show()
    #
    # # print(np.maximum(mask,))
    # # 将连接起来的区域对应的数组和原图对应位置按位相与
    # masked = cv2.bitwise_and(image, image, mask=im)
    # # plt.figure()
    # # plt.imshow(masked)
    # # plt.show()
    # ### 裁剪图片，先把目标的外接矩形裁出来，再扩展到统一尺寸crop_size
    # ### 小的图片直接用0填充，大的图片缩放到该尺寸
    # # 目标外接正矩形
    # crop_masked = masked[left_down[1]:right_up[1], left_down[0]:right_up[0], :]
    # plt.figure()
    # plt.imshow(crop_masked)
    # # plt.show()
    #
    # ### 摆正图片（需要标注时从目标固定方位开始标注）
    # rbox2 = poly_to_rotated_box_single(list(polygon.reshape(-1))) # 转为rbox，查看旋转角度
    # angle =  np.rad2deg(float(rbox2[-1]))
    # print(rbox2)
    # # crop_masked = image_rotate(crop_masked,angle,int(rbox2[2]),int(rbox2[3]),)
    # crop_masked = rotate_bound(crop_masked, angle, border_value= (255,255,255))
    # plt.figure()
    # plt.imshow(crop_masked)
    # plt.show()

    ### 缩放图片
    # 获取目标尺寸
    hw = np.array(img_crop.shape[:2])
    # 判断目标大小是否大于crop_size
    if max(hw) > crop_size:
        # 缩放
        zoom_ratio = crop_size/max(hw)
        hw = hw * zoom_ratio
        hw[np.argmax(hw)] = crop_size

        img_crop = cv2.resize(img_crop,(int(hw[1]),int(hw[0])),0,0)
        # 不够的地方再填充

        # plt.figure()
        # plt.imshow(crop_masked)
        # plt.show()
    # 填充图片到（crop_size,cropsize）
    expand_top = int((crop_size-img_crop.shape[0])/2)
    expand_bottom = crop_size - img_crop.shape[0] - expand_top
    expand_left = int((crop_size-img_crop.shape[1])/2)
    expand_right = crop_size - img_crop.shape[1] - expand_left
    img_crop = cv2.copyMakeBorder(img_crop, expand_top, expand_bottom, expand_left, expand_right, cv2.BORDER_CONSTANT,
                             value=(0, 0, 0))
    assert img_crop.shape[:2] == (crop_size, crop_size)

    # plt.imshow(img_crop)
    # plt.show()

    cv2.imwrite(save_file_path, img_crop)


# def img_paste2():
#     rotateImage


def extract_object(txt_dir,img_dir,classes, ext=".png"):
    """
    # 从大图中提取目标，然后保存成小切片
    :param txt_dir:
    :param img_dir:
    :param classes:
    :param ext:
    :return:
    """
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

    img_dir = "/home/jyc/arashi/data/HRSC2016/FullDataSet/AllImages"
    txt_dir = os.path.join(img_dir,"../labelTxt")
    classes = get_cls_list("/home/jyc/arashi/data/HRSC2016/FullDataSet/classes.txt")
    # classes = ["企业级航母","尼米兹级航母","俄罗斯库兹涅佐夫号航母","中途号航母"]
    classes = ["阿利伯克级驱逐舰","圣安东尼奥级两栖船坞运输舰","塔拉瓦级通用两栖攻击舰","琵琶形军舰","提康德罗加级巡洋舰","佩里级护卫舰","尾部OX头部圆指挥舰",
             "奥斯汀级两栖船坞运输舰","惠德贝岛级船坞登陆舰","医疗船"]
    extract_object(txt_dir, img_dir, classes, ext=".bmp")