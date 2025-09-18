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
            '20240518_204055',
            '20240518_204146',
            '20240518_204240',
            '20240518_204334',
            '20240518_204428',
            '20240518_204525',
            '20240518_204623',
            '20240518_204723',
            '20240518_204823',
            '20240518_204921',
            '20240518_205023',
            '20240518_205123',
            '20240518_205226',
            '20240518_205331',
            '20240518_205434',
            '20240518_205538',
            '20240518_205621',
            '20240518_205728',
            '20240518_205837']
    
    u_inftys = []
    u_rel_l2s = []
    ranks = []
    for TIME_STR in TIME_STRs:
        path = '/' + TASK_NAME + '/' + TIME_STR + '/'

        # 加载模型
        net_path = root_path + '/' + path + '/PINN.pkl'
        model_config = PINNConfig.reload_config(net_path=net_path)
        
        x_valid = model_config.x
        u_true = model_config.funU(x_valid)
        
        us = model_config.net_model(x_valid)
        us = model_config.ortho_us(us)
        u_pred = torch.matmul(us, model_config.W)

        
        ubs = model_config.net_model(model_config.xb)
        ubs = model_config.reload(ubs, requires_grad=False)
        Lus = model_config.A
        A = torch.cat((Lus, ubs), dim=0)

        rank = np.max(model_config.detach(torch.linalg.matrix_rank(A)))
        # log('rank ' + str(rank))
        ranks.append(rank)

        u_true = model_config.detach(u_true)
        u_pred = model_config.detach(u_pred)
        x_valid = model_config.detach(x_valid)

        errors = np.abs(u_true - u_pred)

        
        # ubs = model_config.net_model(model_config.xb)
        # Lus = model_config.A
        # A = torch.cat((Lus, ubs), dim=0)

        # detach_A = model_config.detach(A)
        # rank = np.linalg.matrix_rank(detach_A)
        # ranks.append(rank)

        scipy.io.savemat(root_path + '/' +
                        TASK_NAME + '/' + TIME_STR + '/data.mat', {'u_pred': u_pred, 'u_true': u_true, 'x_valid': x_valid, 'errors': errors})

        u_infty = np.max(np.linalg.norm(u_true-u_pred, ord=inf))
        u_rel_l2 = np.max(np.linalg.norm(u_true-u_pred)/np.linalg.norm(u_true))
        log_str = 'u_infty %.4E u_rel_l2 %.4E' % (u_infty, u_rel_l2)
        log(log_str)
        u_inftys.append(u_infty)
        u_rel_l2s.append(u_rel_l2)

    file_name = root_path + '/' + TASK_NAME + '/error_helmholtz_1D'
    datas = []
    data_labels = []
    layer_sizes = [220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
    # ranks = [220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
    # ranks = [220, 230, 240, 220, 242, 270, 277, 257, 290, 271, 317, 318, 312, 293, 251, 306, 310, 346, 289]

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
    
    
    file_name = file_name + '_rank'
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
        u_true = model_config.funU(x_valid)
        
        us = model_config.net_model(x_valid)
        us = model_config.ortho_us(us)
        u_pred = torch.matmul(us, model_config.W)

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