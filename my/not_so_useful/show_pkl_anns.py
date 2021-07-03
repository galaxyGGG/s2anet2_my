# 检查转换为pkl后的rotated box是否正确
import pickle
from matplotlib import pyplot as plt
from my.DOTA import DOTA
from mmdet.core import rotated_box_to_poly_single
import os
from my.data_process.convert_fair1m_to_mmdet import wordname_15 as classnames

path = '/home/jyc/arashi/data/FAIR1M_dataset_small_1024/trainval_split/trainval_s2anet.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
f = open(path, 'rb')
data = pickle.load(f)

for img in data:
    img_name = img["filename"]
    dict_ann = img["ann"]
    rboxes = dict_ann["bboxes"]
    bboxes = []

    labels = dict_ann["labels"]
    anns = []
    for i in range(len(labels)):
        dict_temp = {}
        rbox = rboxes[i]
        bbox = rotated_box_to_poly_single(rbox)
        label_name = classnames[labels[i]]
        dict_temp["name"] = label_name
        dict_temp["difficult"] = 0
        dict_temp["poly"] = [(bbox[0],bbox[1]),(bbox[2],bbox[3]),(bbox[4],bbox[5]),(bbox[6],bbox[7])]
        dict_temp["area"] = 0
        anns.append(dict_temp)

    examplesplit = DOTA(os.pardir(path))
    imgids = examplesplit.getImgIds()
    for imgid in imgids:
        # anns = examplesplit.loadAnns(imgId=imgid)
        examplesplit.showAnns(anns, imgid, 2)
        plt.show()

