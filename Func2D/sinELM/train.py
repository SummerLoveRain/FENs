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
    
    TRAIN_GRID_SIZE = 100
    X = np.linspace(lb[0], ub[0], TRAIN_GRID_SIZE+1)
    Y = np.linspace(lb[1], ub[1], TRAIN_GRID_SIZE+1)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X_TRAIN, Y_TRAIN = np.meshgrid(X, Y)
    
    X_Train = np.hstack((X_TRAIN.flatten()[:, None], Y_TRAIN.flatten()[:, None]))
    x = X_Train[:, 0:1]
    y = X_Train[:, 1:2]
        
    VALID_GRID_SIZE = 200
    X = np.linspace(lb[0], ub[0], VALID_GRID_SIZE+1)
    Y = np.linspace(lb[1], ub[1], VALID_GRID_SIZE+1)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X_VALID, Y_VALID = np.meshgrid(X, Y)
    
    X_Valid = np.hstack((X_VALID.flatten()[:, None], Y_VALID.flatten()[:, None]))
    x_valid = X_Valid[:, 0:1]
    y_valid = X_Valid[:, 1:2]

    # x = x_valid
    # y = y_valid

    # x_valid = x
    # y_valid = y

    # N_basis = 400
    # scale = 7.7
    # N_basis = 900
    # scale = 9.1
    # N_basis = 1600
    # scale = 10.3
    N_basis = 2500
    scale = 22.7


    in_num = 2
    out_num = N_basis
    layer_num = N_basis
    
    # 打印参数
    log_str = 'TRAIN_GRID_SIZE '+str(TRAIN_GRID_SIZE) + ' VALID_GRID_SIZE '+str(VALID_GRID_SIZE) + ' N_basis ' + str(N_basis)
    log(log_str)

    # 训练参数
    train_dict = {
        'x': x,
        'y': y,
        'x_valid': x_valid,
        'y_valid': y_valid,
        'funU': funU,
        'N_basis': N_basis,
    }

    # block_layers = [layer_num, layer_num]
    # block_num = 2
    # model_name = MODEL_NAME.ResNet
    # model_dict = {
    #     'in_num': in_num,
    #     'out_num': out_num,
    #     'block_layers': block_layers,
    #     'block_num': block_num
    # }

    layers = [in_num, out_num]
    # layers = [in_num, layer_num, layer_num, layer_num, out_num]
    model_name = MODEL_NAME.sinELM
    model_dict = {
        'layers': layers,
        'scale': scale,
        'init_type': INIT_TYPE.Uniform,
        'init_params':{
            'var': 1
        }
    }

    log(str(model_dict))
    # train_Adam_LBFGS(model_dict, model_name, device, param_dict, train_dict, Adam_steps=50000, Adam_init_lr=1e-3, LBFGS_steps=50000)
    train_Adam(model_dict, model_name, device, param_dict, train_dict, Adam_steps=1)