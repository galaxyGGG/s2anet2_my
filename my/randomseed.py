"""
随机数的种子是随机数初始值，种子确定了则随机数不变
"""
import numpy as np

num = 0
while (num < 5):
    np.random.seed(0)
    print(np.random.rand(1, 5))
    num += 1

print('-------------------------')