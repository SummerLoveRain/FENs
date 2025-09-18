import numpy as np

### 训练或者预测可能需要的参数

a = 0
b = 8
# 设置定义域
lb = np.array([a])
ub = np.array([b])

lamb = 50
beta = 10

def funU(x, type='torch'):
    if type == 'torch':
        import torch
        pkg = torch
    else:
        pkg = np
    sin = pkg.sin
    cos = pkg.cos
    pi = pkg.pi

    u = sin(3*pi*x+3*pi*0.05)*cos(4*pi*x-2*pi*0.2) + 1.5 + 0.1*x
    # u = sin(3*pi*x)
    return u


def funF(lamb, beta, x, type='torch'):
    if type == 'torch':
        import torch
        pkg = torch
    else:
        pkg = np
    sin = pkg.sin
    cos = pkg.cos
    pi = pkg.pi

    u = funU(x, type)
    sin_u = sin(u)
    f = -24*pi**2*cos(3*pi*x+3*pi*0.05)*sin(4*pi*x-2*pi*0.2) \
        -25*pi**2*sin(3*pi*x+3*pi*0.05)*cos(4*pi*x-2*pi*0.2) \
        - lamb*u + beta*sin_u
    # f = -(9*pi**2+lamb)*u + beta*sin_u
    return f

