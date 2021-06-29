# Author: Arashi
# Time: 2021/6/28 上午8:39
# Desc:  把imgagenet格式的数据由小类合并为大类

import os
import shutil


def merge_cat(data_dir,tar_dir):
    cats = os.listdir(data_dir) # 所有类别目录
    if not os.path.exists(tar_dir):
        os.mkdir(tar_dir)
    for cat in cats:
        cat_dir = os.path.join(data_dir,cat)
        cat_new = cat
        # 判断该类别属于哪个类
        if cat in ["船","游轮","运输汽车船(======|","货船(_|.--.--|_]=","气垫船","运输汽车船([]==[])","集装箱货船","商船","游艇"]:
            cat_new = "ship"
        elif cat in ["阿利伯克级驱逐舰","圣安东尼奥级两栖船坞运输舰","塔拉瓦级通用两栖攻击舰","琵琶形军舰","提康德罗加级巡洋舰","佩里级护卫舰","尾部OX头部圆指挥舰",
             "奥斯汀级两栖船坞运输舰","惠德贝岛级船坞登陆舰","军舰","医疗船","蓝岭级指挥舰",]:
            cat_new = "warship"
        elif cat in ["企业级航母","尼米兹级航母","俄罗斯库兹涅佐夫号航母","中途号航母","小鹰级航母","航母"]:
            cat_new = "aircraft_carrier"
        elif cat in ["潜艇"]:
            cat_new ="submarine"
        cat_new_dir = os.path.join(tar_dir,cat_new)
        if not os.path.exists(cat_new_dir):
            os.mkdir(cat_new_dir)

        for img in os.listdir(cat_dir):
            img_path = os.path.join(cat_dir,img)
            img_to_path = os.path.join(cat_new_dir,img)
            shutil.copyfile(img_path, img_to_path)

if __name__ == '__main__':
    data_dir = "/home/jyc/arashi/data/HRSC2016/FullDataSet/extracted_obj"
    tar_dir = "/home/jyc/arashi/data/HRSC2016/FullDataSet/extracted_obj_category"
    merge_cat(data_dir,tar_dir)
    print(tar_dir)






















