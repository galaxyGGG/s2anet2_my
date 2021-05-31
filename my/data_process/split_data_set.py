#!/usr/bin/env python
# coding=utf-8

import shutil

import numpy as np
import os
from tqdm import tqdm


def split_dataset(data_dir, train_ratio, val_ratio, out_dir):
    """
    # 划分数据集，分别放到对应的文件夹
    # 要先把标注文件转换为txt格式
    :param data_dir:
    :param train_ratio:
    :param val_ratio:
    :param out_dir:
    :return:
    """
    data = os.listdir(data_dir)
    # 设置随机数种子，保证每次生成的结果都是一样的
    np.random.seed(42)
    # permutation随机生成0-len(data)随机序列
    shuffled_indices = np.random.permutation(len(data))

    # 训练集
    train_set_size = int(len(data) * train_ratio)
    train_indices = shuffled_indices[:train_set_size]
    # 验证集
    val_set_size = int(len(data) * val_ratio)
    val_indices = shuffled_indices[train_set_size:(train_set_size+val_set_size)]
    # 测试集
    test_indices = shuffled_indices[(train_set_size+val_set_size):]

    # 移动文件
    for i,set_name in enumerate(["train", "val", "test"]):
        print(i,set_name)
        img_dir = os.path.join(out_dir,set_name,"images")
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        label_dir = os.path.join(out_dir, set_name, "labelTxt")
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        for indice in tqdm([train_indices, val_indices, test_indices][i]):
            filename = data[indice]
            # 复制image
            file_path = os.path.join(data_dir,filename)
            file_out_path = os.path.join(img_dir,filename)
            shutil.copyfile(file_path, file_out_path)
            # 复制label
            txt_name = os.path.splitext(filename)[0]+".txt"
            label_path = os.path.abspath(os.path.join(data_dir, "..", "labelTxt", txt_name))
            label_out_path = os.path.join(label_dir, txt_name)
            shutil.copyfile(label_path, label_out_path)





if __name__ == '__main__':
    data_dir = "/home/jyc/arashi/data/FAIR1M/images_png"
    out_dir = "/home/jyc/arashi/data/FAIR1M_dataset"

    split_dataset(data_dir, 0.7, 0.1,out_dir)

