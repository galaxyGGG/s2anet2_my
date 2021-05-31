from PIL import Image
im = Image.open("/home/jyc/arashi/data/FAIR1M/images/1390.tif")
print(im.getbands())