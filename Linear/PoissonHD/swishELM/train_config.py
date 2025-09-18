import numpy as np
import torch

### 训练或者预测可能需要的参数

# 设置维度
d = 15
# 设置定义域
lb = np.array([-1 for i in range(d)])
ub = np.array([1 for i in range(d)])


def funF(*args):
    sin = torch.sin
    
    X = torch.concatenate(*args, dim=1)
    d = X.shape[1]
    sum_d_x = torch.sum(X, dim=1, keepdim=True)/d

    f = (sin(sum_d_x)-2)/d
    return f

def funU(*args):
    sin = torch.sin
    
    X = torch.concatenate(*args, dim=1)
    d = X.shape[1]
    sum_d_x = torch.sum(X, dim=1, keepdim=True)/d

    u = sum_d_x**2 + sin(sum_d_x)
    return u
