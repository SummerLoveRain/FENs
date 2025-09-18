import os
import os.path
import datetime
import logging
import os
import random
import numpy as np
import torch

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
    if len(argv) > 1 and 'cuda' == argv[1] and torch.cuda.is_available(
    ):
        device = 'cuda'
    else:
        device = 'cpu'
    print('using device ' + device)
    logging.info('using device ' + device)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    # device = torch.device('cpu')
    return device