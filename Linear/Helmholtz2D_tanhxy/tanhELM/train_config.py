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
    tanh = pkg.tanh

    q = (k**2-2*x**2-2*y**2)*tanh(x*y) + (2*x**2+2*y**2)*tanh(x*y)**3
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
    tanh = pkg.tanh

    u = tanh(x*y)
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
    tanh = pkg.tanh

    h = tanh(x*y)
    return h

