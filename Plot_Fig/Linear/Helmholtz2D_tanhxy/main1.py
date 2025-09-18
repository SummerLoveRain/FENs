u_inftys_cosELM = [3.1021464563796286e-10, 1.2337353361147052e-14, 3.885780586188048e-15, 5.995204332975845e-15]
u_rel_l2_cosELM = [1.3276245746816459e-10, 6.772816107465813e-15, 3.3389877381755397e-15, 5.2278425822747906e-15]
u_inftys_csELM = [4.093105854252599e-10, 1.2101430968414206e-14, 3.2612801348363973e-15, 3.7192471324942744e-15]
u_rel_l2_csELM = [1.692867849785796e-10, 9.68046132143919e-15, 2.4688712255284023e-15, 2.247275716492243e-15]
u_inftys_sinELM = [1.0436331798757692e-09, 1.6042722705833512e-14, 4.107825191113079e-15, 4.218847493575595e-15]
u_rel_l2_sinELM = [4.3592266836328965e-10, 8.761732229106267e-15, 3.845396949951139e-15, 3.903889568525002e-15]
u_inftys_sigmoidELM = [2.9184219041411552e-08, 2.549849220656597e-11, 2.495781359357352e-13, 2.0039525594484076e-14]
u_rel_l2_sigmoidELM = [1.591581644066525e-08, 1.5287342943193466e-11, 1.558782640484268e-13, 2.1435962095849805e-14]
u_inftys_swishELM = [2.5342513421122703e-08, 1.776778724149608e-11, 3.4794389591752406e-13, 3.84969833788773e-14]
u_rel_l2_swishELM = [1.0114670233828291e-08, 1.8566267509992178e-11, 3.964143269718151e-13, 5.837257098491685e-14]
u_inftys_tanhELM = [2.3647467317644555e-08, 1.602540322664936e-11, 1.8673951274195133e-13, 1.176836406102666e-14]
u_rel_l2_tanhELM = [1.2001139326108807e-08, 1.01722922386437e-11, 1.532448671345574e-13, 9.180599528940365e-15]


import numpy as np
basis = [400, 900, 1600, 2500]

datas = []
data = np.stack((basis, u_inftys_cosELM), 1)
datas.append(data)
data = np.stack((basis, u_inftys_sinELM), 1)
datas.append(data)
data = np.stack((basis, u_inftys_csELM), 1)
datas.append(data)
data = np.stack((basis, u_inftys_sigmoidELM), 1)
datas.append(data)
data = np.stack((basis, u_inftys_swishELM), 1)
datas.append(data)
data = np.stack((basis, u_inftys_tanhELM), 1)
datas.append(data)

data_labels =[]
data_labels.append('cos')
data_labels.append('sin')
data_labels.append('cos & sin')
data_labels.append('sigmoid')
data_labels.append('swish')
data_labels.append('tanh')

root_path = '/home/lunxu/yangqh/FEN/Plot_Fig/Linear/Helmholtz2D_3'
file_name =  root_path + '/error_Helmholtz2D'
xy_labels = ['M', 'error']
from plot.line import plot_line
plot_line(datas,
              data_labels,
              xy_labels,
              title = None,
              file_name=file_name,
              xlog=False,
              ylog=True,
              marker=True,
              linestyle=True)