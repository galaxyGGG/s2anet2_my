# Author: Arashi
# Time: 2021/7/2 下午6:38
# Desc:  检测大类+分类识别小类后，对得到的txt文件进行评价，计算map


import os
from DOTA_devkit.dota_evaluation_task1 import voc_eval
import numpy as np

def my_eval(dst_merge_path,gt_dir,imagesetfile,classes):
    """
    计算map
    :param dst_merge_path:
    :param gt_dir:
    :param classes:
    :return:
    """
    detpath = os.path.join(dst_merge_path, '{:s}.txt')
    annopath = os.path.join(gt_dir, '{:s}.txt')

    classaps = []
    map = 0
    for classname in classes:
        rec, prec, ap = voc_eval(detpath,
                                 annopath,
                                 imagesetfile,
                                 classname,
                                 ovthresh=0.5,
                                 use_07_metric=True)
        map = map + ap
        print(classname, ': ', ap)
        classaps.append(ap)

    map = map / len(classes)
    print('map:', map)
    classaps = 100 * np.array(classaps)
    print('classaps: ', classaps)

if __name__ == '__main__':
    dst_merge_path = "/home/jyc/arashi/PycharmProjects/s2anet2_my/work_dirs/s2anet_r50_fpn_1x_dota_hrsc2016_category/merged_cls"
    gt_dir="/home/jyc/arashi/data/HRSC2016_dataset/test/labelTxt"
    imagesetfile  ="/home/jyc/arashi/data/HRSC2016_dataset/test/imgsetfile.txt"
    classes = ['ship', 'medic', 'burke', 'perry', 'talawa', 'submarine',
                      "圣安东尼奥级两栖船坞运输舰", "琵琶形军舰", "提康德罗加级巡洋舰", "尾部OX头部圆指挥舰",
                     "奥斯汀级两栖船坞运输舰", "惠德贝岛级船坞登陆舰",
                      "企业级航母", "尼米兹级航母", "俄罗斯库兹涅佐夫号航母", "中途号航母"
                      ]
    my_eval(dst_merge_path, gt_dir, imagesetfile, classes)