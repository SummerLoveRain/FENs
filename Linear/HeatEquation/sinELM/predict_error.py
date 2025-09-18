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
        '20240518_063837',
        '20240518_063901',
        '20240518_063935',
        '20240518_064042',
        '20240518_064204',
        '20240518_064330',
        '20240518_064518',
        '20240518_064808',
        '20240518_065134',
        '20240518_065554',
        '20240518_070105',
        '20240518_070727',
        '20240518_071452',
        '20240518_072344',
        '20240518_073406',
        '20240518_074630',
        '20240518_080100',
        '20240518_081742',
        '20240518_083739',
        '20240518_090034',
        '20240518_092647',
        '20240518_095555',
        '20240518_102855']
    
    u_inftys = []
    u_rel_l2s = []
    u_inftys1 = []
    u_rel_l2s1 = []
    ranks = []
    for TIME_STR in TIME_STRs:
        path = '/' + TASK_NAME + '/' + TIME_STR + '/'

        # 加载模型
        net_path = root_path + '/' + path + '/PINN.pkl'
        model_config = PINNConfig.reload_config(net_path=net_path)
        
        x_valid = model_config.x
        y_valid = model_config.y
        t_valid = model_config.t
        u_true = model_config.funU(x_valid, y_valid, t_valid)
        
        us = model_config.net_model(x_valid, y_valid, t_valid)


        x = model_config.x
        y = model_config.y
        t = model_config.t
        xb = model_config.xb
        yb = model_config.yb
        tb = model_config.tb
        x0 = model_config.x0
        y0 = model_config.y0
        t0 = model_config.t0
        f = model_config.funF(x, y, t)
        h = model_config.funU(xb, yb, tb)
        q = model_config.funU(x0, y0, t0)
        ubs = model_config.net_model(xb, yb, tb)
        u0s = model_config.net_model(x0, y0, t0)
        ubs = model_config.reload(ubs, requires_grad=False)
        u0s = model_config.reload(u0s, requires_grad=False)
        
        ubs = model_config.net_model(xb, yb, tb)
        u0s = model_config.net_model(x0, y0, t0)
        ubs = model_config.reload(ubs, requires_grad=False)
        u0s = model_config.reload(u0s, requires_grad=False)

        Lus = model_config.A
        A = torch.cat((Lus, ubs, u0s), dim=0)
        b = torch.cat((f, h, q), dim=0)

        rank = np.max(model_config.detach(torch.linalg.matrix_rank(A)))
        # log('rank ' + str(rank))
        ranks.append(rank)

        # AT = torch.transpose(A, dim0=0, dim1=1)
        # W = torch.matmul(torch.linalg.pinv(torch.matmul(AT, A)), torch.matmul(AT, b))
        # W = model_config.lstsq(A=A, b=b)
        # W = torch.matmul(torch.linalg.pinv(A), b)
        # W = torch.matmul(torch.linalg.inv(torch.matmul(AT, A)), torch.matmul(AT, b))

        
        A = model_config.detach(A)
        b = model_config.detach(b)
        W = np.matmul(np.linalg.pinv(A), b)
        W1 = np.linalg.lstsq(A, b)[0]
        print('max error:', np.max(W-W1))
        W = model_config.data_loader(W, requires_grad=False)
        
        
        u_pred = torch.matmul(us, W)

        u_true = model_config.detach(u_true)
        u_pred = model_config.detach(u_pred)

        u_infty = np.max(np.linalg.norm(u_true-u_pred, ord=inf))
        u_rel_l2 = np.max(np.linalg.norm(u_true-u_pred)/np.linalg.norm(u_true))
        # log_str = 'u_infty %.4E u_rel_l2 %.4E' % (u_infty, u_rel_l2)
        # log(log_str)
        u_inftys1.append(u_infty)
        u_rel_l2s1.append(u_rel_l2)

        us = model_config.ortho_us(us)
        u_pred = torch.matmul(us, model_config.W)

        u_pred = model_config.detach(u_pred)
        x_valid = model_config.detach(x_valid)
        y_valid = model_config.detach(y_valid)
        t_valid = model_config.detach(t_valid)

        errors = np.abs(u_true - u_pred)

        # ubs = model_config.net_model(model_config.xb, model_config.yb, model_config.tb)
        # u0s = model_config.net_model(model_config.x0, model_config.y0, model_config.t0)
        
        # Lus = model_config.A
        # A = torch.cat((Lus, ubs, u0s), dim=0)
        # detach_A = model_config.detach(A)
        # rank = np.linalg.matrix_rank(detach_A)
        # ranks.append(rank)

        scipy.io.savemat(root_path + '/' +
                        TASK_NAME + '/' + TIME_STR + '/data.mat', {'u_pred': u_pred, 'u_true': u_true, 'x_valid': x_valid, 'y_valid': y_valid, 't_valid':t_valid, 'errors': errors})

        u_infty = np.max(np.linalg.norm(u_true-u_pred, ord=inf))
        u_rel_l2 = np.max(np.linalg.norm(u_true-u_pred)/np.linalg.norm(u_true))
        # log_str = 'u_infty ' + str(u_infty) + ' u_rel_l2 ' + str(u_rel_l2)
        log_str = 'u_infty %.4E u_rel_l2 %.4E' % (u_infty, u_rel_l2)
        log(log_str)
        u_inftys.append(u_infty)
        u_rel_l2s.append(u_rel_l2)

    file_name = root_path + '/' + TASK_NAME + '/error_heat'
    datas = []
    data_labels = []
    layer_sizes = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]
    # ranks = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]
    data_labels.append('$L_{\infty}$ of OrthoNet')
    data_labels.append('$L_{2}$ of OrthoNet')
    data_labels.append('$L_{\infty}$ of LSTSQ')
    data_labels.append('$L_{2}$ of LSTSQ')
    data = np.stack((layer_sizes, u_inftys), 1)
    datas.append(data)
    data = np.stack((layer_sizes, u_rel_l2s), 1)
    datas.append(data)
    data = np.stack((layer_sizes, u_inftys1), 1)
    datas.append(data)
    data = np.stack((layer_sizes, u_rel_l2s1), 1)
    datas.append(data)

    datas2 = []
    data_labels2 = []
    data = np.stack((layer_sizes, ranks), 1)
    datas2.append(data)
    data_labels2.append('$rank$')
    
    print('rank ' + str(ranks))
    
    xy_labels = ['M', 'error', 'rank']
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
        y_valid = model_config.y_valid
        t_valid = model_config.t_valid
        u_true = model_config.funU(x_valid, y_valid, t_valid)
        
        us = model_config.net_model(x_valid, y_valid, t_valid)
        us = model_config.ortho_us(us)
        u_pred = torch.matmul(us, model_config.W)

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