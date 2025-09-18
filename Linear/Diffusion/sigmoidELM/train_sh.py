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
    
    T_GRID_SIZE = 100
    T = np.linspace(lb[1], ub[1], T_GRID_SIZE+1)
    T = T[1:]

    TRAIN_GRID_SIZE = 100
    X = np.linspace(lb[0], ub[0], TRAIN_GRID_SIZE+1)

    X1 = X[1:-1]
    X1 = np.asarray(X1, dtype=float)
    x1 = np.reshape(X1, (X1.shape[0], 1))
    
    x0 = np.reshape(X, (X.shape[0], 1))
    t0 = 0*x0 + lb[1]
    
    xb = None
    tb = None
    x = None
    t = None
    xb0 = [[a1], [b1]]
    xb0 = np.asarray(xb0)
    xb0 = np.reshape(xb0, (xb0.shape[0], 1))
    for ti in T:        
        if xb is None:
            xb = xb0
            tb = 0*xb0 + ti

            x = x1
            t = 0*x1 + ti
        else:
            xb = np.concatenate((xb, xb0), axis=0)
            tb = np.concatenate((tb, 0*xb0 + ti), axis=0)
            
            x = np.concatenate((x, x1), axis=0)
            t = np.concatenate((t, 0*x1 + ti), axis=0)
        
    VALID_GRID_SIZE = 1000
    X = np.linspace(lb[0], ub[0], VALID_GRID_SIZE+1)
    X = np.asarray(X, dtype=float)
    
    x_valid = np.reshape(X, (X.shape[0], 1))
    t_valid = 0*x_valid+ub[1]
    # t_valid = 0*x_valid+10

    N_basis = int(sys.argv[2])
    if N_basis is None:        
        ValueError("The N_basis is not correct!")
    in_num = 2
    out_num = N_basis
    layer_num = N_basis
    
    # 打印参数
    log_str = 'T_GRID_SIZE ' + str(T_GRID_SIZE)+ ' TRAIN_GRID_SIZE '+str(TRAIN_GRID_SIZE) + ' VALID_GRID_SIZE '+str(VALID_GRID_SIZE) + ' N_basis ' + str(N_basis) + ' tf ' + str(tf)
    log(log_str)

    # 训练参数
    train_dict = {
        'nu': nu,
        'x': x,
        't': t,
        'xb': xb,
        'tb': tb,
        'x0': x0,
        't0': t0,
        'x_valid': x_valid,
        't_valid': t_valid,
        'funF': funF,
        'funU': funU,
        'N_basis': N_basis,
    }

    layers = [in_num, out_num]
    # layers = [in_num, layer_num, layer_num, layer_num, out_num]
    model_name = MODEL_NAME.FourierNet
    model_dict = {
        'layers': layers,
        # 'init_type': INIT_TYPE.Xavier_Normal,
        # 'init_type': INIT_TYPE.Xavier_Uniform,
        # 'init_type': INIT_TYPE.Kaiming_Normal,
        # 'init_type': INIT_TYPE.Kaiming_Uniform,
        'init_type': INIT_TYPE.Qihong_Normal,
        'init_params':{
            'var_x': 0.5
        },
        # 'init_type': INIT_TYPE.Qihong_Uniform,
        # 'init_params':{
        #     'var_x': 0.5
        # }
    }
    log(model_dict)
    # train_Adam_LBFGS(model_dict, model_name, device, param_dict, train_dict, Adam_steps=50000, Adam_init_lr=1e-3, LBFGS_steps=50000)
    train_Adam(model_dict, model_name, device, param_dict, train_dict, Adam_steps=1)