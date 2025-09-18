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
        '20240518_224510',
        '20240518_224551',
        '20240518_224639',
        '20240518_224734',
        '20240518_224841',
        '20240518_225002',
        '20240518_225142',
        '20240518_225347',
        '20240518_225620',
        '20240518_225934',
        '20240518_230327',
        '20240518_230811',
        '20240518_231342',
        '20240518_232032'
    ]
    
    u_inftys = []
    u_rel_l2s = []
    ranks = []
    for TIME_STR in TIME_STRs:
        path = '/' + TASK_NAME + '/' + TIME_STR + '/'

        # 加载模型
        net_path = root_path + '/' + path + '/PINN.pkl'
        model_config = PINNConfig.reload_config(net_path=net_path)
        
        x_valid = model_config.x
        y_valid = model_config.y
        t_valid = model_config.t
        u_true = model_config.funU(x_valid, y_valid, t_valid, model_config.epsilon)
        
        u_pred = model_config.forward(x_valid, y_valid, t_valid)

        u_true = model_config.detach(u_true)
        u_pred = model_config.detach(u_pred)
        x_valid = model_config.detach(x_valid)
        y_valid = model_config.detach(y_valid)
        t_valid = model_config.detach(t_valid)

        errors = np.abs(u_true - u_pred)

        ubs = model_config.net_model(model_config.xb, model_config.yb, model_config.tb)
        u0s = model_config.net_model(model_config.x0, model_config.y0, model_config.t0)
        
        Lus = model_config.A
        A = torch.cat((Lus, ubs, u0s), dim=0)

        rank = np.max(model_config.detach(torch.linalg.matrix_rank(A)))
        # log('rank ' + str(rank))
        ranks.append(rank)

        scipy.io.savemat(root_path + '/' +
                        TASK_NAME + '/' + TIME_STR + '/data.mat', {'u_pred': u_pred, 'u_true': u_true, 'x_valid': x_valid, 'y_valid': y_valid, 't_valid':t_valid, 'errors': errors})

        u_infty = np.max(np.linalg.norm(u_true-u_pred, ord=inf))
        u_rel_l2 = np.max(np.linalg.norm(u_true-u_pred)/np.linalg.norm(u_true))
        # log_str = 'u_infty ' + str(u_infty) + ' u_rel_l2 ' + str(u_rel_l2)
        log_str = 'u_infty %.4E u_rel_l2 %.4E' % (u_infty, u_rel_l2)
        log(log_str)
        u_inftys.append(u_infty)
        u_rel_l2s.append(u_rel_l2)

    file_name = root_path + '/' + TASK_NAME + '/error_burgers'
    datas = []
    data_labels = []
    # layer_sizes = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]
    # ranks = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2099, 2199, 2298, 2397, 2492]
    layer_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]
    # ranks = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]
    data_labels.append('$L_{\infty}$')
    data_labels.append('$L_{2}$')
    data = np.stack((layer_sizes, u_inftys), 1)
    datas.append(data)
    data = np.stack((layer_sizes, u_rel_l2s), 1)
    datas.append(data)

    datas2 = []
    data_labels2 = []
    data = np.stack((layer_sizes, ranks), 1)
    datas2.append(data)
    data_labels2.append('$rank$')
    
    print('rank ' + str(ranks))
    
    
    xy_labels = ['size', 'error', 'rank']
    plot_line(datas,
              data_labels,
              xy_labels,
              title=None,
              file_name=file_name,
              xlog=False,
              ylog=True)
    
    
    file_name = root_path + '/' + TASK_NAME + '/error_burgers_rank'
    plot_line2(datas,
                data_labels,
                datas2,
                data_labels2,
                xy_labels,
                title=None,
                file_name=file_name,
                yscale2=None,
                xlog=False,
                ylog1=True,
                ylog2=False)
            
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
        t_valid = model_config.t_valid
        u_true = model_config.funU(x_valid, y_valid, t_valid, model_config.epsilon)
        
        u_pred = model_config.forward(x_valid, y_valid, t_valid)

        u_true = model_config.detach(u_true)
        u_pred = model_config.detach(u_pred)
        x_valid = model_config.detach(x_valid)
        y_valid = model_config.detach(y_valid)
        t_valid = model_config.detach(t_valid)

        errors = np.abs(u_true - u_pred)

        scipy.io.savemat(root_path + '/' +
                        TASK_NAME + '/' + TIME_STR + '/data.mat', {'u_pred': u_pred, 'u_true': u_true, 'x_valid': x_valid, 'y_valid': y_valid, 't_valid':t_valid, 'errors': errors})

        u_infty = np.max(np.linalg.norm(u_true-u_pred, ord=inf))
        u_rel_l2 = np.max(np.linalg.norm(u_true-u_pred)/np.linalg.norm(u_true))
        # log_str = 'u_infty ' + str(u_infty) + ' u_rel_l2 ' + str(u_rel_l2)
        log_str = 'u_infty %.4E u_rel_l2 %.4E' % (u_infty, u_rel_l2)
        log(log_str)