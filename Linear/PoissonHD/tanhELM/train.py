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
    if d==5:
        N_B = 1000
        N_R = 50000
        # N_basis = 2500
        # scale = 0.028
        N_basis = 10000
        scale = 0.12
    elif d==7:
        N_B = 1000
        N_R = 50000
        # N_basis = 2500
        # scale = 0.0196
        N_basis = 10000
        scale = 0.033
    elif d==10:
        N_B = 1000
        N_R = 50000
        # N_basis = 2500
        # scale = 0.0161
        N_basis = 10000
        scale = 0.035
    elif d==15:
        N_B = 1000
        N_R = 50000
        # N_basis = 2500
        # scale = 0.0002
        N_basis = 10000
        scale = 0.024
    else:
        N_B = 1000
        N_R = 50000
        N_basis = 2500
    # lhs采样 size=[2,N_f]
    x = lb + (ub-lb)*lhs(d, N_R)

    x_b = []
    if d==1:
        x_b.append(lb[0])
        x_b.append(ub[0])
        x_b = np.array(x_b).reshape(2, 1)
    else:
        for i in range(d):
            x_bi =  lb[i] + (ub[i]-lb[i])*lhs(d-1, N_B)
            x_bi0 = np.insert(x_bi, i, lb[i], axis=1)
            x_bi1 = np.insert(x_bi, i, ub[i], axis=1)
            x_b.append(x_bi0)
            x_b.append(x_bi1)
        x_b = np.array(x_b).reshape(N_B * d * 2, d)

    N_basis = N_basis
    in_num = d
    out_num = N_basis
    layer_num = N_basis
    
    # 打印参数
    log_str = 'd ' + str(d) + ' N_R '+str(N_R) + ' N_B '+str(N_B) + ' N_basis ' + str(N_basis)
    log(log_str)

    # 训练参数
    train_dict = {
        'x': x,
        'x_b': x_b,
        'funF': funF,
        'funU': funU,
        'N_basis': N_basis,
    }


    layers = [in_num, out_num]
    # layers = [in_num, layer_num, layer_num, layer_num, out_num]
    model_name = MODEL_NAME.tanhELM
    model_dict = {
        'layers': layers,
        'scale': scale,
        'init_type': INIT_TYPE.Uniform,
        'init_params':{
            'var': 1
        },
    }

    log(str(model_dict))
    # train_Adam_LBFGS(model_dict, model_name, device, param_dict, train_dict, Adam_steps=50000, Adam_init_lr=1e-3, LBFGS_steps=50000)
    train_Adam(model_dict, model_name, device, param_dict, train_dict, Adam_steps=1)