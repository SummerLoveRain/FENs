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


    TIME_STRs = [
            '20241220_161853']
    
    for TIME_STR in TIME_STRs:
        path = '/' + TASK_NAME + '/' + TIME_STR + '/'

        # 加载模型
        net_path = root_path + '/' + path + '/PINN.pkl'
        model_config = PINNConfig.reload_config(net_path=net_path)
        
        x_valid = model_config.x
        # x_valid = model_config.x_valid
        u_true = model_config.funU(x_valid)
        # u_pred = model_config.forward(x_valid)
        
        u_pred = model_config.forward(x_valid)

        u_true = model_config.detach(u_true)
        u_pred = model_config.detach(u_pred)
        x_valid = model_config.detach(x_valid)

        errors = np.abs(u_true - u_pred)

        scipy.io.savemat(root_path + '/' +
                        TASK_NAME + '/' + TIME_STR + '/data.mat', {'u_pred': u_pred, 'u_true': u_true, 'x_valid': x_valid, 'errors': errors})

        u_infty = np.max(np.linalg.norm(u_true-u_pred, ord=inf))
        u_rel_l2 = np.max(np.linalg.norm(u_true-u_pred)/np.linalg.norm(u_true))
        log_str = 'u_infty %.4E u_rel_l2 %.4E' % (u_infty, u_rel_l2)
        log(log_str)

        file_name = root_path + '/' + TASK_NAME + '/' + TIME_STR + '/helmholtz_1D'
        datas1 = []
        datas2 = []
        data_labels1 = []
        data_labels2 = []
        data_labels1.append('$u_{true}$')
        data_labels1.append('$u_{pred}$')
        data_labels2.append('error')
        data = np.stack((x_valid, u_true), 1)
        datas1.append(data)
        data = np.stack((x_valid, u_pred), 1)
        datas1.append(data)
        data = np.stack((x_valid, errors), 1)
        datas2.append(data)

        # xy_labels = ['x', '$sin(3\pi x)$', 'error']
        # xy_labels = ['x', r"$sin(3 \pi x + \frac{3 \pi}{20}) cos(4 \pi x- \frac{2 \pi}{5}) + \frac{3}{2} + \frac{x}{10}$", 'error']
        xy_labels = ['x', r"$u$", 'error']

        # plot_line2(datas1,
        #         data_labels1,
        #         datas2,
        #         data_labels2,
        #         xy_labels,
        #         title=None,
        #         file_name=file_name,
        #         yscale2=None,
        #         xlog=False,
        #         ylog1=False,
        #         ylog2=True)
        
        plot_line(datas1,
              data_labels1,
              xy_labels,
              title=None,
              file_name=file_name,
              xlog=False,
              ylog=False)
                
    elapsed = time.time() - start_time
    print('Predicting time: %.4f' % (elapsed))
