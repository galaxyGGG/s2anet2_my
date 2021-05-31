import os
import cv2
from tqdm import tqdm


def Convert_To_Png(dir):
    """
    # 转换tif为png
    :param dir:
    :return:
    """
    files = os.listdir(dir)
    ResultPath1 = os.path.abspath(os.path.join(dir,"..","images_png"))  # 定义转换格式后的保存路径
    if not os.path.exists(ResultPath1):
        os.mkdir(ResultPath1)
    for file in tqdm(files):  # 这里可以去掉for循环

        a, b = os.path.splitext(file)  # 拆分影像图的文件名称
        this_dir = os.path.join(dir,file)  # 构建保存 路径+文件名

        img = cv2.imread(this_dir, 1)  # 读取tif影像
        # 第二个参数是通道数和位深的参数，
        # IMREAD_UNCHANGED = -1  # 不进行转化，比如保存为了16位的图片，读取出来仍然为16位。
        # IMREAD_GRAYSCALE = 0  # 进行转化为灰度图，比如保存为了16位的图片，读取出来为8位，类型为CV_8UC1。
        # IMREAD_COLOR = 1   # 进行转化为RGB三通道图像，图像深度转为8位
        # IMREAD_ANYDEPTH = 2  # 保持图像深度不变，进行转化为灰度图。
        # IMREAD_ANYCOLOR = 4  # 若图像通道数小于等于3，则保持原通道数不变；若通道数大于3则只取取前三个通道。图像深度转为8位
        try:
            cv2.imwrite(os.path.join(ResultPath1, a +".png"), img)  # 保存为png格式
        except:
            print(file + "\n")

if __name__ == '__main__':
    # img = cv2.imread(r"/home/jyc/arashi/data/FAIR1M/images/1390.tif", 1)
    # cv2.imwrite(r"/home/jyc/arashi/data/FAIR1M/1390.png", img)
    Convert_To_Png("/home/jyc/arashi/data/FAIR1M/images")