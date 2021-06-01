import pickle
path = '/home/jyc/arashi/PycharmProjects/s2anet2_my/tools/work_dirs/s2anet_r50_fpn_1x_fair1m/res.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f)

print(data)
print(len(data))
