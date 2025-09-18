import numpy as np
import torch

### 训练或者预测可能需要的参数

# 设置定义域
lb = np.array([0, 0])
ub = np.array([1, 1])

# epsilon = 1.0
# epsilon = 0.5
# epsilon = 0.2
# epsilon = 0.1
epsilon = 0.01

def funT(x, t):
    sin = torch.sin
    T = sin((x-t)/(2*epsilon))
    return T

def funT_x(x, t):
    sin = torch.sin
    cos = torch.cos
    T = cos((x-t)/(2*epsilon))
    T_x = 1/(2*epsilon)*T
    return T_x

def funT_xx(x, t):
    sin = torch.sin
    T = -sin((x-t)/(2*epsilon))
    T_xx = 1/(4*epsilon**2)*T
    return T_xx

def funT_t(x, t):
    cos = torch.cos
    T = cos((x-t)/(2*epsilon))
    T_t = -1/(2*epsilon)*T
    return T_t
    


def funU(x, t):
    exp = torch.exp
    u = 1/(1 + funT(x, t))
    return u

def funU_t(x, t):
    T = funT(x, t)
    T_t = funT_t(x, t)
    u_t = -1/(1 + T)**2*T_t
    return u_t

def funU_x(x, t):
    T = funT(x, t)
    T_x = funT_x(x, t)
    u_x = -1/(1 + T)**2*T_x
    return u_x

def funU_xx(x, t):
    T = funT(x, t)
    T_x = funT_x(x, t)
    T_xx = funT_xx(x, t)
    u_xx = 2/(1+T)**3 * T_x**2 - 1/(1+T)**2 * T_xx
    return u_xx

def funF(x, t):
    u = funU(x, t)
    u_t = funU_t(x, t)
    u_x = funU_x(x, t)
    u_xx = funU_xx(x, t)
    f = u_t + u*u_x - epsilon*u_xx
    return f


