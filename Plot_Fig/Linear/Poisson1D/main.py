u_inftys_cosELM = [3.0724E-11, 1.3878E-11, 7.2102E-11, 1.7749E-11]
u_rel_l2_cosELM = [9.7238E-11, 3.0395E-11, 9.4554E-11, 3.2850E-11]
u_inftys_FourierNet = [1.9459E-11, 6.1339E-11, 1.7186E-11, 1.9275E-11]
u_rel_l2_FourierNet = [3.0421E-11, 1.1078E-10, 2.8037E-11, 3.1528E-11]
u_inftys_sinELM = [3.2458E-11, 2.9600E-11, 2.7607E-10, 4.6233E-11]
u_rel_l2_sinELM = [5.7302E-11, 4.9456E-11, 4.0441E-10, 9.1841E-11]
u_inftys_sigmoidELM = [3.6288E-05, 2.9000E-07, 1.6587E-07, 1.0117E-06]
u_rel_l2_sigmoidELM = [3.4848E-05, 6.0888E-07, 1.8036E-07, 1.0275E-06]
u_inftys_swishELM = [1.7748E-04, 1.4265E-06, 5.9915E-07, 1.9759E-06]
u_rel_l2_swishELM = [2.4055E-04, 1.5708E-06, 8.1065E-07, 2.1461E-06]
u_inftys_tanhELM = [3.0062E-05, 2.6801E-07, 6.1572E-08, 2.0924E-07]
u_rel_l2_tanhELM = [3.8163E-07, 2.8331E-07, 8.0396E-08, 1.7937E-07]

import numpy as np
basis = [400, 900, 1600, 2500]

datas = []
data = np.stack((basis, u_inftys_cosELM), 1)
datas.append(data)
data = np.stack((basis, u_inftys_sinELM), 1)
datas.append(data)
data = np.stack((basis, u_inftys_FourierNet), 1)
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

root_path = '/home/dell/yangqh/FEN/Plot_Fig/Linear/Poisson1D'
file_name =  root_path + '/error_Poisson1D'
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