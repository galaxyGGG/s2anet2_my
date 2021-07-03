# 生成imgsetfile文件，做evaluation的时候要用

import os

def write_img_names(dir,with_ext=False):
    files = os.listdir(dir)
    txt_path = os.path.join(dir,"..","imgsetfile.txt")
    with open(txt_path,"w") as f:
        for file in files:
            # 不加后缀
            if not with_ext:
                f.write(os.path.splitext(file)[0]+"\n")
            else:
                # 加后缀
                f.write(file + "\n")



if __name__ == '__main__':
    write_img_names("/home/jyc/arashi/data/HRSC2016_dataset/test/images",False)