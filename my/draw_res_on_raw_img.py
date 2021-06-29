import os
import random

import cv2
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset

colormap=[
        (54, 67, 244),
        (99, 30, 233),
        (176, 39, 156),
        (183, 58, 103),
        (181, 81, 63),
        (243, 150, 33),
        (212, 188, 0),
        (136, 150, 0),
        (80, 175, 76),
        (74, 195, 139),
        (57, 220, 205),
        (59, 235, 255),
        (0, 152, 255),
        (34, 87, 255),
        (72, 85, 121)]
#随机生成颜色
from my.get_colors import ncolors
colormap = ncolors(5, 0)

def draw_res_on_raw_img(img_dir,img_out_dir,txt_dir,config_file,show_label,score_thd=0.3):
    cfg = Config.fromfile(config_file)
    data_test = cfg.data.test
    dataset = build_dataset(data_test)
    class_names = dataset.CLASSES

    img_list = os.listdir(img_dir)

    for i in range(len(img_list)):
        img_name = img_list[i]
        img_path = os.path.join(img_dir, img_name)
        if colormap:
            assert len(class_names) == len(colormap)
        img = mmcv.imread(img_path)
        color_white = (255, 255, 255)

        res_list = []
        # 获取所有标注信息

        for j, name in enumerate(class_names):
            if colormap:
                color = colormap[j]
            else:
                color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))

            txt_path = os.path.join(txt_dir,name+".txt")
            with open(txt_path,"r") as f:
                lines = f.readlines()
                for line in lines:
                    ann_list = line.strip().split(" ")
                    if not ann_list[0] == os.path.splitext(img_name)[0]:
                        continue
                    score = float(ann_list[1])
                    if score<score_thd:
                        continue
                    bbox = list(map(int,map(float,ann_list[2:])))
                    for i in range(3):
                        cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i + 1) * 2], bbox[(i + 1) * 2 + 1]),
                                 color=color,
                                 thickness=2, lineType=cv2.LINE_AA)
                    cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2,
                             lineType=cv2.LINE_AA)
                    if show_label:
                        cv2.putText(img, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                                    color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
        if not os.path.exists(img_out_dir):
            os.makedirs(img_out_dir)
        img_out_path = os.path.join(img_out_dir,img_name)
        cv2.imwrite(img_out_path, img)
        print("finish:",img_name)


if __name__ == '__main__':
    img_dir = "/home/amax/ganlan/arashi/data/FAIR1M_dataset/test/images"
    img_out_dir = "/home/amax/ganlan/arashi/data/FAIR1M_dataset/test/infer_imgs"
    txt_dir = "/home/amax/ganlan/arashi/s2anet2_my/s2anet_r50_fpn_1x_scale1_1.5_fair1m_5_classes/results_after_nms"
    config_file = "/home/amax/ganlan/arashi/s2anet2_my/my/config/s2anet_r50_fpn_1x_fair1m_5classes.py"
    draw_res_on_raw_img(img_dir, img_out_dir, txt_dir, config_file, True,0.5)