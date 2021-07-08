#!/usr/bin/env python
# coding=utf-8

import shutil

import numpy as np
import os
from tqdm import tqdm


def split_dataset_dota(data_dir, train_ratio, val_ratio, out_dir):
    """
    # 划分数据集，分别放到对应的文件夹
    # 要先把标注文件转换为txt格式
    一般来说用测试集做验证集，所以val_ratio = 0
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
            # *** 复制image ***
            file_path = os.path.join(data_dir,filename)
            file_out_path = os.path.join(img_dir,filename)
            shutil.copyfile(file_path, file_out_path)
            # *** 复制label ***
            txt_name = os.path.splitext(filename)[0]+".txt"
            label_path = os.path.abspath(os.path.join(data_dir, "..", "labelTxt", txt_name))
            label_out_path = os.path.join(label_dir, txt_name)
            shutil.copyfile(label_path, label_out_path)


def split_dataset_imagenet(data_dir, train_ratio, val_ratio, out_dir):
    """
    imagenet的数据按照类别建立文件夹，把每个文件夹分别按照划分比例分为训练、验证集
    一般来说用测试集做验证集，所以val_ratio = 0
    :param data_dir:
    :param train_ratio:
    :param val_ratio:
    :param out_dir:
    :return:
    """
    cls_names = os.listdir(data_dir)
    # 设置随机数种子，保证每次生成的结果都是一样的
    np.random.seed(42)

    for cls_name in cls_names:
        # 类别文件夹
        cls_dir = os.path.join(data_dir,cls_name)

        data = os.listdir(cls_dir)
        # permutation随机生成0-len(data)随机序列
        shuffled_indices = np.random.permutation(len(data))

        # 训练集
        train_set_size = int(len(data) * train_ratio)
        train_indices = shuffled_indices[:train_set_size]
        # 验证集
        val_set_size = int(len(data) * val_ratio)
        val_indices = shuffled_indices[train_set_size:(train_set_size + val_set_size)]
        # 测试集
        test_indices = shuffled_indices[(train_set_size + val_set_size):]

        # 移动文件
        for i, set_name in enumerate(["train", "val", "test"]):
            print(i, set_name)
            out_cls_dir = os.path.join(out_dir, set_name,  cls_name)
            if not os.path.exists(out_cls_dir):
                os.makedirs(out_cls_dir)
            for indice in tqdm([train_indices, val_indices, test_indices][i]):
                filename = data[indice]
                # *** 复制image ***
                file_path = os.path.join(cls_dir, filename)
                file_out_path = os.path.join(out_cls_dir, filename)
                shutil.copyfile(file_path, file_out_path)


if __name__ == '__main__':
    # # *** dota 数据集划分 ***
    # data_dir = "/home/jyc/arashi/data/HRSC2016/FullDataSet/AllImages"
    # out_dir = "/home/jyc/arashi/data/HRSC2016_dataset"
    # split_dataset_dota(data_dir, 0.8, 0, out_dir)

    # *** image_net数据集划分 ***
    data_dir = "/home/jyc/arashi/data/HRSC2016_cls/all/images"
    out_dir = "/home/jyc/arashi/data/HRSC2016_cls/all"
    split_dataset_imagenet(data_dir, 0.9, 0, out_dir)

