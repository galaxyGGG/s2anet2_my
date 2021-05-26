import os
dir="/home/jyc/下载/s2anet/work_dirs/upload"
res=os.listdir(dir)
for i in res:
 if ".txt" in i:
    os.rename(os.path.join(dir,i),os.path.join(dir,'Task1_'+i))