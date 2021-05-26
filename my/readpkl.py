import pickle
path = '/data/dota_1024/trainval_split/trainval_s2anet.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f)

print(data)
print(len(data))
