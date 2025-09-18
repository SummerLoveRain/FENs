import numpy as np

### 训练或者预测可能需要的参数

# 设置定义域
lb = np.array([-1, -1])
ub = np.array([1, 1])

a1 = 1
a2 = 4
k = 1

def funQ(a1, a2, k, x, y, type='torch'):
    if type == 'torch':
        import torch
        pkg = torch
    else:
        pkg = np
    sin = pkg.sin
    cos = pkg.cos
    pi = pkg.pi

    q = -(a1*pi)**2*sin(a1*pi*x)*sin(a2*pi*y) - (a2*pi)**2*sin(a1*pi*x)*sin(a2*pi*y) + k**2*sin(a1*pi*x)*sin(a2*pi*y) 
    return q

def funU(a1, a2, k, x, y, type='torch'):
    if type == 'torch':
        import torch
        pkg = torch
    else:
        pkg = np
    sin = pkg.sin
    cos = pkg.cos
    pi = pkg.pi

    u = sin(a1*pi*x)*sin(a2*pi*y) 
    return u


def funH(a1, a2, k, x, y, type='torch'):
    if type == 'torch':
        import torch
        pkg = torch
    else:
        pkg = np
    sin = pkg.sin
    cos = pkg.cos
    pi = pkg.pi

    h = sin(a1*pi*x)*sin(a2*pi*y) 
    return h

