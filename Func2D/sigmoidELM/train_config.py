import numpy as np
import torch

### 训练或者预测可能需要的参数

# 设置定义域
lb = np.array([-1, -1])
ub = np.array([1, 1])

a1 = 1
a2 = 4

def funU(x, y, type='torch'):
    if type == 'torch':
        import torch
        pkg = torch
    else:
        pkg = np
    sin = pkg.sin
    cos = pkg.cos
    pi = pkg.pi
    exp = pkg.exp

    u = sin(a1*pi*x)*sin(a2*pi*y)
    return u
