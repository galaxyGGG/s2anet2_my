import os
import shutil


# 在数据集中找到含有某些特定的类别的数据
def find_by_cls(cls_list, txt_dir, img_dir, output_dir):
    list = os.listdir(txt_dir)  # 列出文件夹下所有的目录与文件
    for name in list:
        path = os.path.join(txt_dir, name)
        if not os.path.isfile(path):
            continue
        if not path.endswith(".txt"):
            continue
        with open(path) as f:
            content = f.readlines()
            count_cls = 0
            if len(content)-2 ==0:
                 continue
            for i in range(2,len(content)):
                line = content[i].split()
                if line[-2] in cls_list:
                    # print(line)
                    count_cls += 1
            if count_cls/(len(content)-2) >= 0.4:
                img_path = os.path.join(img_dir,name.split(".txt")[0]+".png")
                new_img_dir = os.path.join(output_dir,"images")
                if not os.path.exists(new_img_dir):
                    os.mkdir(new_img_dir)
                new_img_path = os.path.join(new_img_dir,name.split(".txt")[0]+".png")
                shutil.copyfile(img_path,new_img_path)
                # 还要移动txt
                txt_path = os.path.join(txt_dir, name)
                new_txt_dir = os.path.join(output_dir, "labelTxt")
                if not os.path.exists(new_txt_dir):
                    os.mkdir(new_txt_dir)
                new_txt_path = os.path.join(new_txt_dir, name)
                shutil.copyfile(txt_path, new_txt_path)


if __name__ == '__main__':

    # # DOTA有现成的函数
    # examplesplit = DOTA('/home/jyc/下载/s2anet/data/dota/train')
    # imgids = examplesplit.getImgIds(catNms=['ship'])
    # img = examplesplit.loadImgs(imgids)
    # for imgid in imgids:


    cls_list = ["ship"]
    root_path = "/home/jyc/下载/s2anet/data/dota/train/labelTxt"
    img_path = "/home/jyc/下载/s2anet/data/dota/train/images"
    output_path = "/home/jyc/下载/s2anet/data/ganlan/ganlan_ship"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    find_by_cls(cls_list,root_path,img_path,output_path)


