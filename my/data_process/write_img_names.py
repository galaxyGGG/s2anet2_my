import os


def write_img_names(dir):
    files = os.listdir(dir)
    txt_path = os.path.join(dir,"..","imgsetfile.txt")
    with open(txt_path,"w") as f:
        for file in files:
            f.write(os.path.splitext(file)[0]+"\n")

if __name__ == '__main__':
    write_img_names("/home/amax/ganlan/arashi/data/FAIR1M_dataset/test/images")