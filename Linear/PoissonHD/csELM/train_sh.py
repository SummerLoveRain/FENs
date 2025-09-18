import logging
import sys
from pyDOE import lhs
import torch
import torch.nn as nn   
from model import MODEL_NAME
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
    
    TRAIN_GRID_SIZE = 50
    X = np.linspace(lb[0], ub[0], TRAIN_GRID_SIZE+1)
    Y = np.linspace(lb[1], ub[1], TRAIN_GRID_SIZE+1)
    # 处理边界条件
    xb_D = X
    xb_D = np.reshape(xb_D, (xb_D.shape[0], 1))
    yb1_D = 0*xb_D + lb[1]
    yb2_D = 0*xb_D + ub[1]
    xb_D = np.concatenate((xb_D, xb_D), axis=0)
    yb_D = np.concatenate((yb1_D, yb2_D), axis=0)

    yb_N = Y[1:-1]
    yb_N = np.reshape(yb_N, (yb_N.shape[0], 1))
    xb1_N = 0*yb_N + lb[0]
    xb2_N = 0*yb_N + ub[0]
    n1 = 0*yb_N - 1
    n2 = 0*yb_N + 1
    n = np.concatenate((n1, n2), axis=0)
    xb_N = np.concatenate((xb1_N, xb2_N), axis=0)
    yb_N = np.concatenate((yb_N, yb_N), axis=0)

    X = X[1:-1]
    Y = Y[1:-1]
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X_TRAIN, Y_TRAIN = np.meshgrid(X, Y)
    
    X_Train = np.hstack((X_TRAIN.flatten()[:, None], Y_TRAIN.flatten()[:, None]))
    x = X_Train[:, 0:1]
    y = X_Train[:, 1:2]
        
    VALID_GRID_SIZE = 100
    X = np.linspace(lb[0], ub[0], VALID_GRID_SIZE+1)
    Y = np.linspace(lb[1], ub[1], VALID_GRID_SIZE+1)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X_VALID, Y_VALID = np.meshgrid(X, Y)
    
    X_Valid = np.hstack((X_VALID.flatten()[:, None], Y_VALID.flatten()[:, None]))
    x_valid = X_Valid[:, 0:1]
    y_valid = X_Valid[:, 1:2]

    N_basis = int(sys.argv[2])
    if N_basis is None:        
        ValueError("The N_basis is not correct!")

    N_basis_x = N_basis
    N_basis_y = N_basis
    in_num = 1
    out_num_x = N_basis_x
    out_num_y = N_basis_y
    layer_num_x = N_basis_x
    layer_num_y = N_basis_y
    
    # 打印参数
    log_str = 'TRAIN_GRID_SIZE '+str(TRAIN_GRID_SIZE) + ' VALID_GRID_SIZE '+str(VALID_GRID_SIZE) + ' N_basis_x ' + str(N_basis_x) + ' N_basis_y ' + str(N_basis_y)
    log(log_str)


    # 训练参数
    train_dict = {
        'x': x,
        'y': y,
        'xb_D': xb_D,
        'yb_D': yb_D,
        'xb_N': xb_N,
        'yb_N': yb_N,
        'n': n,
        'funF': funF,
        'funU': funU,
        'x_valid': x_valid,
        'y_valid': y_valid,
        'N_basis_x': N_basis_x,
        'N_basis_y': N_basis_y,
    }

    block_layers_x = [layer_num_x, layer_num_x]
    block_layers_y = [layer_num_y, layer_num_y]
    block_num_x = 1
    block_num_y = 1
    model_name = MODEL_NAME.MultiModels
    model_dict = {
        'model_dicts': [
            {
                'in_num': in_num,
                'out_num': out_num_x,
                'block_layers': block_layers_x,
                'block_num': block_num_x
            },
            {
                'in_num': in_num,
                'out_num': out_num_y,
                'block_layers': block_layers_y,
                'block_num': block_num_y
            }
        ],
        'model_names':[
            MODEL_NAME.ResNet,
            MODEL_NAME.ResNet
        ]
    }
    log(str(model_dict))
    # train_Adam_LBFGS(model_dict, model_name, device, param_dict, train_dict, Adam_steps=50000, Adam_init_lr=1e-3, LBFGS_steps=50000)
    train_Adam(model_dict, model_name, device, param_dict, train_dict, Adam_steps=1)