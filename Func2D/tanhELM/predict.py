from cmath import inf
import logging
import sys
import time
import numpy as np
from scipy.interpolate import griddata
import torch
from plot.heatmap import plot_heatmap, plot_heatmap3
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

    TIME_STRs = [
                '20240801_193816']

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
        u_true = model_config.funU(x_valid, y_valid)
        
        u_pred = model_config.forward(x_valid, y_valid)

        u_true = model_config.detach(u_true)
        u_pred = model_config.detach(u_pred)
        x_valid = model_config.detach(x_valid)
        y_valid = model_config.detach(y_valid)

        errors = np.abs(u_true - u_pred)

        scipy.io.savemat(root_path + '/' +
                        TASK_NAME + '/' + TIME_STR + '/data.mat', {'u_pred': u_pred, 'u_true': u_true, 'x_valid': x_valid, 'y_valid': y_valid, 'errors': errors})

        u_infty = np.max(np.linalg.norm(u_true-u_pred, ord=inf))
        u_rel_l2 = np.max(np.linalg.norm(u_true-u_pred)/np.linalg.norm(u_true))
        log_str = 'u_infty %.4E u_rel_l2 %.4E' % (u_infty, u_rel_l2)
        log(log_str)

        U_star = griddata(X_Train, u_true.flatten(), (X_TRAIN, Y_TRAIN), method='cubic')
        U_pred = griddata(X_Train, u_pred.flatten(), (X_TRAIN, Y_TRAIN), method='cubic')
        file_name = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/2d_heatmap3'
        # plot_heatmap3(X, Y, T, P, E=None, xlabel=None, ylabel=None, T_title=None, P_title=None, E_title=None, file_name=None, abs=True):
        plot_heatmap3(X=X_TRAIN, Y=Y_TRAIN, T=U_star, P=U_pred, E=None, xlabel='x',
                    ylabel='y', file_name=file_name)

        elapsed = time.time() - start_time
        print('Predicting time: %.4f' % (elapsed))
