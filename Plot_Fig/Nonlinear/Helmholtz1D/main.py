u_inftys_cosELM = [2.9873881146613712e-12, 2.2426505097428162e-13, 9.237055564881302e-13, 7.083222897108499e-13]
u_rel_l2_cosELM = [1.554621608045711e-13, 1.3705170853602015e-14, 4.8288953398025223e-14, 3.577314912843099e-14]
u_inftys_FourierNet = [7.998046669399628e-13, 5.062616992290714e-14, 1.4304113449270517e-12, 1.474376176702208e-13]
u_rel_l2_FourierNet = [3.5969887329033226e-14, 4.186921242937839e-15, 7.064653300547591e-14, 7.875786615712399e-15]
u_inftys_sinELM = [8.664180484174722e-13, 8.462119893692943e-13, 1.4064305275951483e-12, 7.44737604918555e-13]
u_rel_l2_sinELM = [4.226433906050337e-14, 4.64485131640607e-14, 6.468387313687297e-14, 3.744961457290304e-14]
u_inftys_sigmoidELM = [0.00034384214325999807, 5.2436347379725134e-06, 1.5164817943613684e-06, 1.459035814388443e-06]
u_rel_l2_sigmoidELM = [1.6943333335791214e-05, 2.473196859777523e-07, 7.151042204568355e-08, 7.703014121171431e-08]
u_inftys_swishELM = [0.0003172448928796534, 9.288722716327413e-06, 1.412469685657669e-06, 1.4329587822992096e-06]
u_rel_l2_swishELM = [1.941138330312127e-05, 5.510233887627541e-07, 6.678463907284777e-08, 8.116576117404901e-08]
u_inftys_tanhELM = [2.044150666069555e-06, 1.0125980942632395e-08, 3.7985614653734956e-11, 5.633102873048301e-10]
u_rel_l2_tanhELM = [1.0491785322475124e-07, 4.702516747286507e-10, 4.4912057189822014e-12, 3.1178544401053494e-11]

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

root_path = '/home/lunxu/yangqh/FEN/Plot_Fig/Nonlinear/Helmholtz1D'
file_name =  root_path + '/error_Helmholtz1D'
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