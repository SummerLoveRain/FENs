import numpy as np
import torch

### 训练或者预测可能需要的参数

# 设置定义域
lb = np.array([0])
ub = np.array([1])

s = 6

def funF(x):
    u_xx = funU_xx(x)
    f = -u_xx
    return f

def funU(x):
    u = 0
    for i in range(1, s+1):
        u += torch.sin(2**i*torch.pi*x)
        
    u /= s
    return u

def funU_x(x):
    u_x = 0
    for i in range(1, s+1):
        u_x += torch.cos(2**i*torch.pi*x) * 2**i*torch.pi
        
    u_x /= s
    return u_x


def funU_xx(x):
    u_xx = 0
    for i in range(1, s+1):
        u_xx -= torch.sin(2**i*torch.pi*x) * 2**(2*i)*torch.pi**2
        
    u_xx /= s
    return u_xx