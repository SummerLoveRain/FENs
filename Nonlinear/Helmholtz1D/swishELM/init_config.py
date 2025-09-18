import os
import os.path
import datetime
import logging
import os
import random
import time
import numpy as np
import torch
from model import get_model
from model_config import PINNConfig

def getCurDirName():
    '''
    获取当前目录名
    '''
    curfilePath = os.path.abspath(__file__)

    # this will return current directory in which python file resides.
    curDir = os.path.abspath(os.path.join(curfilePath, os.pardir))

    # curDirName = curDir.split(parentDir)[-1]
    curDirName = os.path.split(curDir)[-1]
    return curDirName

def getParentDir():
    '''
    获取父目录路径
    '''
    curfilePath = os.path.abspath(__file__)

    # this will return current directory in which python file resides.
    curDir = os.path.abspath(os.path.join(curfilePath, os.pardir))

    # this will return parent directory.
    parentDir = os.path.abspath(os.path.join(curDir, os.pardir))
    # parentDirName = os.path.split(parentDir)[-1]
    return parentDir


def setup_seed(seed):
    '''
    固定随机种子，让每次运行结果一致
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(0)
TASK_NAME = 'task_' + getCurDirName()
now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
root_path = getParentDir() + '/data/'
path = '/' + TASK_NAME + '/' + now_str + '/'
log_path = root_path + '/' + path


def init_log():
    '''
    开启日志记录功能
    '''
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(filename=os.path.join(log_path, 'log.txt'),
                        level=logging.INFO)


def get_device(argv):
    '''
    从输入获取使用设备
    '''
    if len(argv) > 1 and 'cuda' in argv[1] and torch.cuda.is_available(
    ):
        device = argv[1]
    else:
        device = 'cpu'
    print('using device ' + device)
    logging.info('using device ' + device)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    # device = torch.device('cpu')
    return device

def train_SGD(model_dict, model_name, device, param_dict, train_dict, SGD_steps=50000,  SGD_init_lr=1e-3, scheduler_name=None, scheduler_params=None):
    '''
    用SGD训练
    '''
    # 记录时间
    start_time = time.time()
    # 获取神经网络模型
    model = get_model(model_dict, model_name)
    model.to(device)
    model_config = PINNConfig(param_dict=param_dict,
                              train_dict=train_dict, model=model)
    # 用Adam训练
    if model_config.params is not None:
        params = model_config.params
    else:
        params = model.parameters()
    model_config.train_SGD(params=params, SGD_steps=SGD_steps, SGD_init_lr=SGD_init_lr,
                            scheduler_name=scheduler_name, scheduler_params=scheduler_params)
    # 打印总耗时
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    logging.info('Training time: %.4f' % (elapsed))

def train_Adam(model_dict, model_name, device, param_dict, train_dict, Adam_steps=50000,  Adam_init_lr=1e-3, scheduler_name=None, scheduler_params=None):
    '''
    用Adam训练
    '''
    # 记录时间
    start_time = time.time()
    # 获取神经网络模型
    model = get_model(model_dict, model_name)
    model.to(device)
    model_config = PINNConfig(param_dict=param_dict,
                              train_dict=train_dict, model=model)
    # 用Adam训练
    if model_config.params is not None:
        params = model_config.params
    else:
        params = model.parameters()
    model_config.train_Adam(params=params, Adam_steps=Adam_steps, Adam_init_lr=Adam_init_lr,
                            scheduler_name=scheduler_name, scheduler_params=scheduler_params)
    # 打印总耗时
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    logging.info('Training time: %.4f' % (elapsed))


def train_LBFGS(model_dict, model_name, device, param_dict, train_dict, LBFGS_steps=10000):
    '''
    用LBFGS训练
    '''
    # 记录时间
    start_time = time.time()
    # 获取神经网络模型
    model = get_model(model_dict, model_name)
    model.to(device)
    model_config = PINNConfig(param_dict=param_dict,
                              train_dict=train_dict, model=model)
    # 用LBFGS训练
    if model_config.params is not None:
        params = model_config.params
    else:
        params = model.parameters()
    model_config.train_LBFGS(params=params, LBFGS_steps=LBFGS_steps)

    # 打印总耗时
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    logging.info('Training time: %.4f' % (elapsed))

def train_Adam_LBFGS(model_dict, model_name, device, param_dict, train_dict, Adam_steps=50000, LBFGS_steps=10000):
    '''
    先用Adam训练，再用LBFGS训练
    '''
    # 记录时间
    start_time = time.time()
    # 获取神经网络模型
    model = get_model(model_dict, model_name)
    model.to(device)
    model_config = PINNConfig(param_dict=param_dict,
                              train_dict=train_dict, model=model)
    # 用Adam训练
    if model_config.params is not None:
        params = model_config.params
    else:
        params = model.parameters()
    model_config.train_Adam(params=params, Adam_steps=Adam_steps)
    # 切换训练方式时，应该加载Adam最优结果
    # 加载模型
    net_path = root_path + '/' + path + '/PINN.pkl'
    model_config = PINNConfig.reload_config(net_path=net_path)
    # 用LBFGS训练
    model = model_config.model
    model_config.nIter = 0
    if model_config.params is not None:
        params = model_config.params
    else:
        params = model.parameters()
    model_config.train_LBFGS(params=params, LBFGS_steps=LBFGS_steps)

    # 打印总耗时
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    logging.info('Training time: %.4f' % (elapsed))
