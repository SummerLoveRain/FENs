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


    u_inftys = [9.3441E-05, 6.3137E-05, 4.5144E-04, 1.1430E-07, 1.8614E-07, 3.2485E-06, 2.9161E-08, 2.9069E-09, 6.0161E-08, 1.4853E-09, 1.0044E-06, 7.9083E-10, 1.4887E-10, 7.4220E-09, 5.8241E-10, 5.9058E-08, 1.9526E-09, 2.8819E-10, 1.3299E-10]
    u_rel_l2s = [8.4367E-06, 3.0467E-06, 2.8809E-05, 1.6307E-08, 2.7488E-08, 2.3221E-07, 2.6306E-09, 3.5926E-10, 6.0047E-09, 1.2434E-10, 1.3522E-07, 1.1615E-10, 2.5168E-11, 9.4560E-10, 7.5964E-11, 8.6619E-09, 1.9126E-10, 3.2685E-11, 2.1032E-11]

    file_name = root_path + '/error_helmholtz_1D'
    datas = []
    data_labels = []
    layer_sizes = [220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]
    data_labels.append('$L_{\infty}$ of RRNN')
    # data_labels.append('$L_{2}$ of RRNN')
    data = np.stack((layer_sizes, u_inftys), 1)
    datas.append(data)
    # data = np.stack((layer_sizes, u_rel_l2s), 1)
    # datas.append(data)

    datas2 = []
    data_labels2 = []
    time_NN = [18.8242, 18.6433, 18.9633, 22.9488, 20.4613, 20.1934, 20.8171, 20.4880, 21.3262, 21.3069, 21.8389, 19.1130, 19.9093, 20.4562, 32.8894, 21.7345, 21.2100, 23.3707, 24.1242]
    data = np.stack((layer_sizes, time_NN), 1)
    datas2.append(data)
    data_labels2.append('time of RRNN')
    
    xy_labels = ['M', 'error', 'time']
    plot_line2(datas,
                data_labels,
                datas2,
                data_labels2,
                xy_labels,
                title=None,
                file_name=file_name,
                yscale2=30,
                xlog=False,
                ylog1=True,
                ylog2=False)
            
    elapsed = time.time() - start_time
    print('Predicting time: %.4f' % (elapsed))
