import numpy as np

### 训练或者预测可能需要的参数

# 设置定义域
lb = np.array([0, 0, 0])
ub = np.array([1, 1, 1])


def funF(x, y, t, type='torch'):
    if type == 'torch':
        import torch
        pkg = torch
    else:
        pkg = np
    sin = pkg.sin
    cos = pkg.cos
    exp = pkg.exp
    pi = pkg.pi

    u = funU(x, y, t, type)
    f = (pi**2/4)*u
    return f

def funU(x, y, t, type='torch'):
    if type == 'torch':
        import torch
        pkg = torch
    else:
        pkg = np
    sin = pkg.sin
    cos = pkg.cos
    exp = pkg.exp
    pi = pkg.pi

    u = sin(pi/2*x)*sin(pi/2*y)*sin(pi/2*t)
    return u


def funW(x, y, t, type='torch'):
    if type == 'torch':
        import torch
        pkg = torch
    else:
        pkg = np
    sin = pkg.sin
    cos = pkg.cos
    exp = pkg.exp
    pi = pkg.pi

    w = pi/2*sin(pi/2*x)*sin(pi/2*y)*cos(pi/2*t)
    return w

