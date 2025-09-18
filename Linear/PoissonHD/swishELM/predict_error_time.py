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


    u_inftys = [1.1655E-04, 1.0459E-11, 6.3949E-13, 6.3205E-13, 6.8798E-13]
    u_rel_l2s = [3.5852E-05, 7.4234E-12, 2.6963E-13, 3.5063E-13, 3.9124E-13]
    u_inftys1 = [1.4425E-02, 8.2016E-05, 2.8365E-11, 3.0875E-13, 1.4457E-12]
    u_rel_l2s1 = [5.2671E-03, 1.6428E-05, 1.2661E-11, 2.1826E-13, 5.3645E-13]

    file_name = root_path + '/error_poisson'
    datas = []
    data_labels = []
    layer_sizes = [100, 300, 900, 1600, 2500]
    data_labels.append('$L_{\infty}$ of RRNN')
    # data_labels.append('$L_{2}$ of RRNN')
    data_labels.append('$L_{\infty}$ of TP-RRNN')
    # data_labels.append('$L_{2}$ of TP-RRNN')
    data = np.stack((layer_sizes, u_inftys), 1)
    datas.append(data)
    # data = np.stack((layer_sizes, u_rel_l2s), 1)
    # datas.append(data)
    data = np.stack((layer_sizes, u_inftys1), 1)
    datas.append(data)
    # data = np.stack((layer_sizes, u_rel_l2s1), 1)
    # datas.append(data)

    datas2 = []
    data_labels2 = []
    time_NN = [12.2114, 17.4386, 26.4364, 72.7922, 214.0198]
    time_TP = [11.3960,  11.3970, 11.6845, 11.8703, 11.9978]
    data = np.stack((layer_sizes, time_NN), 1)
    datas2.append(data)
    data_labels2.append('time of RRNN')
    data = np.stack((layer_sizes, time_TP), 1)
    datas2.append(data)
    data_labels2.append('time of TP-RRNN')
    
    xy_labels = ['M', 'error', 'time']
    plot_line2(datas,
                data_labels,
                datas2,
                data_labels2,
                xy_labels,
                title=None,
                file_name=file_name,
                yscale2=220,
                xlog=False,
                ylog1=True,
                ylog2=False)
            
    elapsed = time.time() - start_time
    print('Predicting time: %.4f' % (elapsed))
