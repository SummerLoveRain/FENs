import numpy as np

### 训练或者预测可能需要的参数

a1 = 0
b1 = 5
nu = 0.01
tf = 1

# 设置定义域
lb = np.array([a1, 0])
ub = np.array([b1, tf])

def funF(x, t, nu, type='torch'):
    if type == 'torch':
        import torch
        pkg = torch
    else:
        pkg = np
    sin = pkg.sin
    cos = pkg.cos
    exp = pkg.exp
    pi = pkg.pi

    ux = singleFun(x, type)
    ut = singleFun(t, type)

    u_t = (-2*pi*sin(pi*t+pi/5)-3*pi*sin(2*pi*t-3*pi/5))*ux
    u_xx = (-2*pi**2*cos(pi*x+pi/5)-6*pi**2*cos(2*pi*x-3*pi/5))*ut
    f = u_t-nu*u_xx
    return f

def singleFun(x, type='torch'):
    if type == 'torch':
        import torch
        pkg = torch
    else:
        pkg = np
    sin = pkg.sin
    cos = pkg.cos
    exp = pkg.exp
    pi = pkg.pi
    u = (2*cos(pi*x+pi/5)+3/2*cos(2*pi*x-3*pi/5))
    return u

def funU(x, t, type='torch'):
    if type == 'torch':
        import torch
        pkg = torch
    else:
        pkg = np
    sin = pkg.sin
    cos = pkg.cos
    exp = pkg.exp
    pi = pkg.pi

    u = singleFun(x, type)*singleFun(t, type)
    return u

