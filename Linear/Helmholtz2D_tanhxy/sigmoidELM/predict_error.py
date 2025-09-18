from cmath import inf
import logging
import sys
import time
import numpy as np
from scipy.interpolate import griddata
import torch
from plot.line import plot_density, plot_line, plot_line2
from model_config import PINNConfig

from init_config import TASK_NAME, get_device, path, root_path
from train_config import *
from pyDOE import lhs
import scipy.io

# 打印相关信息
def log(obj):
    print(obj)
    logging.info(obj)

if __name__ == "__main__":
    start_time = time.time()
    device = get_device(sys.argv)

    # 加载各个区域的坐标数据
    TIME_STRs = [
                '20240529_182631',
                '20240529_182708',
                '20240529_182747',
                '20240529_182847',
                '20240529_183106']

    
    u_inftys = []
    u_rel_l2s = []
    u_inftys1 = []
    u_rel_l2s1 = []
    for TIME_STR in TIME_STRs:
        path = '/' + TASK_NAME + '/' + TIME_STR + '/'

        # 加载模型
        net_path = root_path + '/' + path + '/PINN.pkl'
        model_config = PINNConfig.reload_config(net_path=net_path)
        
        TRAIN_GRID_SIZE = 50
        X = np.linspace(lb[0], ub[0], TRAIN_GRID_SIZE+1)
        Y = np.linspace(lb[1], ub[1], TRAIN_GRID_SIZE+1)
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        X_TRAIN, Y_TRAIN = np.meshgrid(X, Y)
        
        X_Train = np.hstack((X_TRAIN.flatten()[:, None], Y_TRAIN.flatten()[:, None]))
        x = X_Train[:, 0:1]
        y = X_Train[:, 1:2]

        x_valid = model_config.data_loader(x, requires_grad = False)
        y_valid = model_config.data_loader(y, requires_grad = False)
        u_true = model_config.funU(model_config.a1, model_config.a2, model_config.k, x_valid, y_valid)
        
        u_pred = model_config.forward(x_valid, y_valid)

        u_true = model_config.detach(u_true)
        u_pred = model_config.detach(u_pred)
        x_valid = model_config.detach(x_valid)
        y_valid = model_config.detach(y_valid)

        errors = np.abs(u_true - u_pred)

        u_infty = np.max(np.linalg.norm(u_true-u_pred, ord=inf))
        u_rel_l2 = np.max(np.linalg.norm(u_true-u_pred)/np.linalg.norm(u_true))
        log_str = 'u_infty %.4E u_rel_l2 %.4E' % (u_infty, u_rel_l2)
        log(log_str)
        u_inftys.append(u_infty)
        u_rel_l2s.append(u_rel_l2)

    file_name = root_path + '/' + TASK_NAME + '/error_helmholtz'
    datas = []
    data_labels = []
    layer_sizes = [100, 300, 900, 1600, 2500]
    data_labels.append('$L_{\infty}$ of RNN')
    data_labels.append('$L_{2}$ of RNN')
    data = np.stack((layer_sizes, u_inftys), 1)
    datas.append(data)
    data = np.stack((layer_sizes, u_rel_l2s), 1)
    datas.append(data)

    
    xy_labels = ['M', 'error']
    plot_line(datas,
              data_labels,
              xy_labels,
              title=None,
              file_name=file_name,
              xlog=False,
              ylog=True)
            
    elapsed = time.time() - start_time
    print('Predicting time: %.4f' % (elapsed))

    print('---------------------')
    for TIME_STR in TIME_STRs:
        path = '/' + TASK_NAME + '/' + TIME_STR + '/'

        # 加载模型
        net_path = root_path + '/' + path + '/PINN.pkl'
        model_config = PINNConfig.reload_config(net_path=net_path)
        
        x_valid = model_config.x_valid
        y_valid = model_config.y_valid
        u_true = model_config.funU(model_config.a1, model_config.a2, model_config.k, x_valid, y_valid)
        
        u_pred = model_config.forward(x_valid, y_valid)

        u_true = model_config.detach(u_true)
        u_pred = model_config.detach(u_pred)
        x_valid = model_config.detach(x_valid)
        y_valid = model_config.detach(y_valid)

        errors = np.abs(u_true - u_pred)

        u_infty = np.max(np.linalg.norm(u_true-u_pred, ord=inf))
        u_rel_l2 = np.max(np.linalg.norm(u_true-u_pred)/np.linalg.norm(u_true))
        log_str = 'u_infty %.4E u_rel_l2 %.4E' % (u_infty, u_rel_l2)
        log(log_str)