import logging
import sys
from pyDOE import lhs
import torch
import torch.nn as nn   
from model import INIT_TYPE, MODEL_NAME
from init_config import *
from train_config import *

# 打印相关信息
def log(obj):
    print(obj)
    logging.info(obj)
    
if __name__ == "__main__":
    # 设置需要写日志
    init_log()
    # cuda 调用
    device = get_device(sys.argv)

    param_dict = {
        'lb': lb,
        'ub': ub,
        'device': device,
        'path': path,
        'root_path': root_path,
    }

    ### 生成训练点 ####
    # 区域内采点数
    N_R = 3000
    # N_R = 100000
    # lhs采样 size=[2,N_f]
    # X = lb + (ub-lb)*lhs(1, N_R)
    # x = X[:, 0:1]

    x = np.linspace(lb[0], ub[0], N_R)
    x = np.reshape(x, (N_R, 1))
    x = x[1:-1, :]
    
    xb = np.concatenate((lb, ub), axis=0)
    xb = np.reshape(xb, (2, 1))
    
    N_Valid = 3000
    x_valid = np.linspace(lb[0], ub[0], N_Valid)
    x_valid = np.reshape(x_valid, (N_Valid, 1))

    # N_basis = 400
    # scale = 9.65
    # N_basis = 900
    # scale = 9.99
    # N_basis = 1600
    # scale = 9.95
    N_basis = 2500
    scale = 9.72

    in_num = 1
    out_num = N_basis
    layer_num = N_basis
    
    # 打印参数
    log_str = 'N_R '+str(N_R) + ' N_Valid ' + str(N_Valid) + ' N_basis ' + str(N_basis)
    log(log_str)


    # 训练参数
    train_dict = {
        'x': x,
        'xb': xb,
        'lamb': lamb,
        'beta': beta,
        'funF': funF,
        'funU': funU,
        'x_valid': x_valid,
        'N_basis': N_basis,
    }

    layers = [in_num, out_num]
    # layers = [in_num, layer_num, layer_num, layer_num, out_num]
    model_name = MODEL_NAME.sigmoidELM    
    model_dict = {
        'layers': layers,
        'scale': scale,
        'init_type': INIT_TYPE.Uniform,
        'init_params':{
            'var': 1
        },
    }
    log(model_dict)
    # train_Adam_LBFGS(model_dict, model_name, device, param_dict, train_dict, Adam_steps=50000, Adam_init_lr=1e-3, LBFGS_steps=50000)
    train_Adam(model_dict, model_name, device, param_dict, train_dict, Adam_steps=1)