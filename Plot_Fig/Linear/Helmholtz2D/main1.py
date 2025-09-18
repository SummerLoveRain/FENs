u_inftys_cosELM = [6.37969860691868e-05, 5.897809264083128e-13, 1.733266333518763e-13, 1.7977229714706125e-13]
u_rel_l2_cosELM = [1.741937923687111e-05, 1.5736409343541276e-13, 7.04011434174587e-14, 7.127258710274106e-14]
u_inftys_csELM = [0.00012551665713544935, 1.5099337692176426e-12, 1.4385444826399062e-13, 3.7487268869077737e-13]
u_rel_l2_csELM = [2.207433233593596e-05, 2.4097677747249355e-13, 6.696402565872466e-14, 1.3203335359967286e-13]
u_inftys_sinELM = [0.00012699265926130465, 9.058700051613472e-13, 1.8736056430121314e-13, 5.3299978373289815e-14]
u_rel_l2_sinELM = [1.8981083932435668e-05, 2.79398867077709e-13, 7.431752808122249e-14, 2.7200119399727913e-14]
u_inftys_sigmoidELM = [0.003387451171875, 6.556510925597526e-07, 8.381903274939656e-09, 6.075424912400435e-10]
u_rel_l2_sigmoidELM = [0.0008315176699687113, 5.72951098313986e-07, 2.8205955046196604e-09, 3.95455734230437e-10]
u_inftys_swishELM = [0.006347656250000103, 7.897615433043815e-07, 6.635673403133094e-09, 1.0986695407438878e-09]
u_rel_l2_swishELM = [0.001576944763830122, 2.2361924980618276e-07, 3.0734858890216385e-09, 3.9584959110521397e-10]
u_inftys_tanhELM = [0.0027465820312499163, 4.4703483593765273e-07, 1.1059455684368162e-08, 4.947652121251306e-10]
u_rel_l2_tanhELM = [0.000677600824971188, 2.0869263468496783e-07, 6.08554152761824e-09, 2.759534252511571e-10]

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

root_path = '/home/lunxu/yangqh/FEN/Plot_Fig/Linear/Helmholtz2D'
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