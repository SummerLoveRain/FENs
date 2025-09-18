u_inftys_cosELM = [1.321222466685225, 0.0004337283462234609, 9.04172101634515e-08, 3.581996921298014e-10]
u_rel_l2_cosELM = [0.06208717109380804, 3.8990888783604796e-05, 1.082780419455971e-08, 4.1604734239936554e-11]
u_inftys_csELM = [1.4927049222290225, 0.0004499044248973405, 1.442938923901238e-08, 6.620792802891629e-11]
u_rel_l2_csELM = [0.059998544548794976, 3.427053290446257e-05, 1.7977362592476856e-09, 5.66911184365354e-12]
u_inftys_sinELM = [1.2850724455633737, 0.002730225947462861, 9.085355667792783e-08, 4.856994806345938e-10]
u_rel_l2_sinELM = [0.05855435401926898, 0.0005029764969174487, 1.1017048826755962e-08, 4.4876418901773277e-11]
u_inftys_sigmoidELM = [13.731721564858457, 2.4171101299219195, 0.07634292373533791, 0.002020761671549076]
u_rel_l2_sigmoidELM = [0.724009439948539, 0.15682359161018997, 0.008592117757303766, 0.00023351708723774175]
u_inftys_swishELM = [16.856830630381225, 6.263641400975729, 0.04354047784195414, 0.0007222724902216093]
u_rel_l2_swishELM = [0.9499283372561232, 0.4165450811728379, 0.0039407221088205, 7.713274793732738e-05]
u_inftys_tanhELM = [14.067618056862852, 4.941664075948464, 0.07829604873533791, 0.0018351825780815734]
u_rel_l2_tanhELM = [0.7370564758729463, 0.4514558364957101, 0.006491395767832013, 0.00017928405186547796]


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

root_path = '/home/lunxu/yangqh/FEN/Plot_Fig/Linear/Diffusion'
file_name =  root_path + '/error_Diffusion_tf10'
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