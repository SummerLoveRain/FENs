import numpy as np
import torch

### 训练或者预测可能需要的参数

# 设置定义域
lb = np.array([0, 0])
ub = np.array([1, 1])

# epsilon = 1.0
# epsilon = 0.5
# epsilon = 0.2
epsilon = 0.1
# epsilon = 0.01

def funT(x, t):
    exp = torch.exp
    T = exp((x-t)/(2*epsilon))
    return T

def funT_x(x, t):
    exp = torch.exp
    T = exp((x-t)/(2*epsilon))
    T_x = 1/(2*epsilon)*T
    return T_x

def funT_xx(x, t):
    exp = torch.exp
    T = exp((x-t)/(2*epsilon))
    T_xx = 1/(4*epsilon**2)*T
    return T_xx

def funT_t(x, t):
    exp = torch.exp
    T = exp((x-t)/(2*epsilon))
    T_t = -1/(2*epsilon)*T
    return T_t
    


def funU(x, t):
    u = 1/torch.sin(1 + funT(x, t))
    return u

def funM(x, t):
    u = 1/(1 + funT(x, t))
    return u

def funM_t(x, t):
    T = funT(x, t)
    T_t = funT_t(x, t)
    M_t = -1/(1 + T)**2*T_t
    return M_t

def funM_x(x, t):
    T = funT(x, t)
    T_x = funT_x(x, t)
    M_x = -1/(1 + T)**2*T_x
    return M_x

def funM_xx(x, t):
    T = funT(x, t)
    T_x = funT_x(x, t)
    T_xx = funT_xx(x, t)
    M_xx = 2/(1+T)**3 * T_x**2 - 1/(1+T)**2 * T_xx
    return M_xx

def funF(x, t):
    M = funM(x, t)
    M_t = funM_t(x, t)
    M_x = funM_x(x, t)
    M_xx = funM_xx(x, t)
    u = torch.sin(M)
    u_t = torch.cos(M)*M_t
    u_x = torch.cos(M)*M_x
    u_xx = -torch.sin(M)*M_x**2 + torch.cos(M)*M_xx
    f = u_t + u*u_x - epsilon*u_xx
    return f


