import argparse
import os
import os.path as osp
import pdb
import random

import cv2
import matplotlib
import mmcv
from mmcv import Config

from mmdet.apis import init_detector, inference_detector,show_result
from mmdet.core import rotated_box_to_poly_single
from mmdet.datasets import build_dataset


def show_result_rbox(img_name,
                     detections,
                     class_names,
                     scale=1.0,
                     threshold=0.2,
                     colormap=None,
                     show_label=False):
    assert isinstance(class_names, (tuple, list))
    if colormap:
        assert len(class_names) == len(colormap)
    img = mmcv.imread(img_name)
    color_white = (255, 255, 255)

    res_list = []

    for j, name in enumerate(class_names):
        if colormap:
            color = colormap[j]
        else:
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        try:
            dets = detections[j]
        except:
            pdb.set_trace()
        # import ipdb;ipdb.set_trace()
        for det in dets:
            score = det[-1]
            det = rotated_box_to_poly_single(det[:-1])
            bbox = det[:8] * scale
            if score < threshold:
                continue
            bbox = list(map(int, bbox))
            # 保存bbox结果,用于后续保存到txt中
            line = osp.splitext(osp.basename(img_name))[0]+" "+str(score)
            for loc in bbox:
                line += " " + str(loc)
            line +=" " + name
            res_list.append(line)

            # 画图
            for i in range(3):
                cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i + 1) * 2], bbox[(i + 1) * 2 + 1]), color=color,
                         thickness=2, lineType=cv2.LINE_AA)
            cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2, lineType=cv2.LINE_AA)
            if show_label:
                cv2.putText(img, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                            color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)

    return img, res_list


def save_det_result(config_file, out_dir, checkpoint_file=None, img_dir=None, colormap=None):
    cfg = Config.fromfile(config_file)
    data_test = cfg.data.test
    dataset = build_dataset(data_test)
    classnames = dataset.CLASSES
    # use checkpoint path in cfg
    if not checkpoint_file:
        checkpoint_file = osp.join(cfg.work_dir, 'latest.pth')
    # use testset in cfg
    if not img_dir:
        img_dir = data_test.img_prefix

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img_list = os.listdir(img_dir)
    for img_name in img_list:
        img_path = osp.join(img_dir, img_name)
        img_out_path = osp.join(out_dir, img_name)
        # print(f"**********************img path is {img_path}")
        result = inference_detector(model, img_path)

        img,bboxes = show_result_rbox(img_path,
                               result,
                               classnames,
                               scale=1.0,
                               threshold=0.5,
                               colormap=colormap,
                              show_label=True)
        print(img_out_path)
        cv2.imwrite(img_out_path, img)
        # 保存bbox结果
        txt_out_dir = osp.abspath(osp.join(out_dir,"..","txt"))
        if not osp.exists(txt_out_dir):
            os.mkdir(txt_out_dir)
        txt_out_path = osp.splitext(osp.join(txt_out_dir,img_name))[0]+".txt"
        with open(txt_out_path,"w") as f:
            for box in bboxes:
                f.write(box+"\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference demo')
    parser.add_argument('config_file', help='input config file')
    parser.add_argument('model', help='pretrain model')
    parser.add_argument('img_dir', help='img dir')
    parser.add_argument('out_dir', help='output dir')
    args = parser.parse_args()

    dota_colormap = [
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
    from my.get_colors import ncolors
    fair1m_colormap = ncolors(37,0)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    hrsc2016_colormap = [(212, 188, 0)]
    save_det_result(args.config_file, args.out_dir, checkpoint_file=args.model, img_dir=args.img_dir,
                    colormap=fair1m_colormap)
