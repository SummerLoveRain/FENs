import logging
import torch
import numpy as np
from torch import autograd
from torch.autograd import Variable
from torch import nn
from typing import List, Tuple

# 打印相关信息
def log(obj):
    print(obj)
    logging.info(obj)

    
def data_loader(x, device='cuda', requires_grad=True):
    '''
    数据加载函数，转变函数类型及其使用设备设置
    '''
    x_tensor = torch.tensor(x,
                            requires_grad=requires_grad,
                            dtype=torch.float64)
    return x_tensor.to(device)

def reload(data, device='cuda', requires_grad=True):
    '''
    重新加载数据
    '''
    data = detach(data)
    data = data_loader(data, device, requires_grad=requires_grad)
    return data


def detach(data):
    '''
    将数据从设备上取出
    '''
    tmp_data = data.detach().cpu().numpy()
    if np.isnan(tmp_data).any():
        raise Exception
    return tmp_data

def triangluar_shift(x, y, vertices):
# 将积分点坐标移动到三角形内部
    x1 = vertices[0, 0]
    y1 = vertices[0, 1]
    x2 = vertices[1, 0]
    y2 = vertices[1, 1]
    x3 = vertices[2, 0]
    y3 = vertices[2, 1]
    Jx1 = x2-x1
    Jx2 = x3-x1
    Jy1 = y2-y1
    Jy2 = y3-y1
    J = Jx1*Jy2 - Jx2*Jy1
    new_x = (Jx1*x + Jx2*y) + x1
    new_y = (Jy1*x + Jy2*y) + y1
    return new_x, new_y, J

# def affine_triangular_xy(x, y, vertices):    
# # 将坐标从全局坐标系变换到三角形坐标系
#     '''
#     x : coordinate
#     y : coordinate
#     vertices: the coordinates of the triangle
#     return : a affine mapping value at (x, y)
#     '''
#     x1 = vertices[0, 0]
#     y1 = vertices[0, 1]
#     x2 = vertices[1, 0]
#     y2 = vertices[1, 1]
#     x3 = vertices[2, 0]
#     y3 = vertices[2, 1]
#     Jx1 = x2-x1
#     Jx2 = x3-x1
#     Jy1 = y2-y1
#     Jy2 = y3-y1
#     J = Jx1*Jy2 - Jx2*Jy1
#     xhat = (Jy2*(x-x1)-Jx2*(y-y1))/J
#     yhat = (-Jy1*(x-x1)+Jx1*(y-y1))/J
#     return xhat, yhat

def affine_triangular_xy(x, y, vertices):
    """
    将坐标从全局坐标系变换到三角形坐标系，支持批量输入
    
    x : 坐标，Tensor 类型，形状为 (N,) 或 (B, N)，其中 B 为批量大小，N 为点数
    y : 坐标，Tensor 类型，形状为 (N,) 或 (B, N)
    vertices: 三角形三个顶点的坐标，Tensor 类型，形状为 (3, 2) 或 (B, 3, 2)
              其中 B 为批量大小，每个三角形包含三个顶点 (x, y)
    return : 仿射映射后的值，形状为 (N,) 或 (B, N)
    """
    # 处理单个三角形或批量三角形
    single_triangle = vertices.dim() == 2
    if single_triangle:
        vertices = vertices.unsqueeze(0)  # (3, 2) -> (1, 3, 2)
    
    # 处理单个点或批量点
    single_point = x.dim() == 1
    if single_point:
        x = x.unsqueeze(0)  # (N,) -> (1, N)
        y = y.unsqueeze(0)  # (N,) -> (1, N)
    
    # 获取三角形顶点坐标
    x1 = vertices[:, 0:1, 0:1]  # (B, 1)
    y1 = vertices[:, 0:1, 1:2]  # (B, 1)
    x2 = vertices[:, 1:2, 0:1]  # (B, 1)
    y2 = vertices[:, 1:2, 1:2]  # (B, 1)
    x3 = vertices[:, 2:3, 0:1]  # (B, 1)
    y3 = vertices[:, 2:3, 1:2]  # (B, 1)
    
    Jx1 = x2 - x1  # (B, 1)
    Jx2 = x3 - x1  # (B, 1)
    Jy1 = y2 - y1  # (B, 1)
    Jy2 = y3 - y1  # (B, 1)
    J = Jx1 * Jy2 - Jx2 * Jy1  # (B, 1)
    
    xhat = (Jy2 * (x - x1) - Jx2 * (y - y1)) / J  # (B, N)
    yhat = (-Jy1 * (x - x1) + Jx1 * (y - y1)) / J  # (B, N)
    
    # 如果是单个三角形和单个点，去掉批量维度
    if single_triangle and single_point:
        xhat = xhat.squeeze(0)  # (1, N) -> (N,)
        yhat = yhat.squeeze(0)  # (1, N) -> (N,)
    elif single_triangle:
        xhat = xhat.squeeze(0)  # (1, N) -> (N,)
        yhat = yhat.squeeze(0)  # (1, N) -> (N,)
    elif single_point:
        xhat = xhat.squeeze(1)  # (B, 1) -> (B,)
        yhat = yhat.squeeze(1)  # (B, 1) -> (B,)
    
    return xhat, yhat

def compute_grad(u, x):
    '''
    直接计算一阶导数
    '''
    # 计算u关于x的一阶导数
    u_x = autograd.grad(u.sum(), x, create_graph=True)[0]
    # 返回u关于x的一阶导数
    return u_x

import torch

def is_in_triangle(x, y, vertices):
    """
    判断点是否在三角形内部（包括边界）
    
    参数:
    x, y -- 待判断点的坐标，Tensor 类型，形状为 (N,)
    vertices -- 三角形三个顶点的坐标，Tensor 类型，形状为 (3, 2) 或 (B, 3, 2)
                其中 B 为批量大小，每个三角形包含三个顶点 (x, y)
    
    返回:
    Tensor -- 布尔类型，形状为 (N,) 或 (B, N)，表示每个点是否在对应的三角形内
    """    
    # 处理单个三角形或批量三角形
    single_triangle = vertices.dim() == 2
    if single_triangle:
        vertices = vertices.unsqueeze(0)  # (3, 2) -> (1, 3, 2)
    
    # 获取三角形顶点坐标
    A = vertices[:, 0]  # (B, 2)
    B = vertices[:, 1]  # (B, 2)
    C = vertices[:, 2]  # (B, 2)
    
    # 计算三角形面积的两倍（用于分母）
    denominator = (B[:, 0:1] - A[:, 0:1]) * (C[:, 1:2] - A[:, 1:2]) - \
                  (B[:, 1:2] - A[:, 1:2]) * (C[:, 0:1] - A[:, 0:1])  # (B,)
    
    # 扩展点坐标以匹配三角形的批量大小
    x_expanded = x.reshape(1, -1).expand(vertices.shape[0], -1)  # (B, N)
    y_expanded = y.reshape(1, -1).expand(vertices.shape[0], -1)  # (B, N)
    
    # 计算重心坐标的三个系数
    alpha_numerator = (B[:, 1:2] - C[:, 1:2]) * (x_expanded - C[:, 0:1]) + \
                      (C[:, 0:1] - B[:, 0:1]) * (y_expanded - C[:, 1:2])  # (B, N)
    beta_numerator = (C[:, 1:2] - A[:, 1:2]) * (x_expanded - C[:, 0:1]) + \
                     (A[:, 0:1] - C[:, 0:1]) * (y_expanded - C[:, 1:2])  # (B, N)
    
    alpha = alpha_numerator / denominator  # (B, N)
    beta = beta_numerator / denominator   # (B, N)
    gamma = 1.0 - alpha - beta                          # (B, N)
    
    # 判断点是否在三角形内部（包括边界）
    eps = 1e-9  # 浮点数容差
    is_inside = (alpha >= -eps) & (beta >= -eps) & (gamma >= -eps)  # (B, N)
    
    # 如果是单个三角形，去掉批量维度
    if single_triangle:
        is_inside = is_inside.squeeze(0)  # (1, N) -> (N,)
    
    return is_inside

def xavier_init(size, device='cuda', requires_grad=True):
    '''
    用xavier随机生成一个可训练的数据
    '''
    W = Variable(nn.init.xavier_normal_(torch.empty(size[0], size[1]))).to(device)
    W.requires_grad_(requires_grad)
    return W

def sample_point_in_triangle() -> Tuple[float, float]:
    """
    在由点 (0,0), (1,0), (0,1) 构成的三角形内随机采样一个点
    
    返回:
        tuple: 包含 x 和 y 坐标的元组 (x, y)
    """
    # 生成两个 0 到 1 之间的随机数
    r1 = np.random.rand()
    r2 = np.random.rand()
    
    # 使用 barycentric 坐标方法生成三角形内的点
    # 确保点位于三角形内的条件: r1 >=0, r2 >=0, r1 + r2 <=1
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2
    
    # 计算点的坐标
    x = r1
    y = r2
    
    return (x, y)

def generate_samples(n: int) -> List[Tuple[float, float]]:
    """
    生成多个随机采样点
    
    参数:
        n (int): 需要生成的采样点数量
    
    返回:
        list: 包含 n 个采样点的列表，每个点表示为 (x, y) 元组
    """
    return [sample_point_in_triangle() for _ in range(n)]