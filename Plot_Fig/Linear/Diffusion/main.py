u_inftys_cosELM = [7.421775971572231e-09, 1.3011813848606835e-13, 1.1723955140041653e-13, 1.8851586958135158e-13]
u_rel_l2_cosELM = [5.565514427823336e-10, 5.943330736463107e-15, 9.143742200528169e-15, 9.145601087895173e-15]
u_inftys_FourierNet = [7.393556877843821e-09, 2.4158453015843406e-13, 1.6608936448392342e-13, 1.2578826869003024e-13]
u_rel_l2_FourierNet = [2.6485262028178637e-10, 1.5478131843833977e-14, 1.3136858643629674e-14, 1.310933845256306e-14]
u_inftys_sinELM = [1.4405212134249723e-08, 9.237055564881302e-14, 1.0302869668521453e-13, 2.113864638886298e-13]
u_rel_l2_sinELM = [8.528575548634463e-10, 5.91863943415533e-15, 8.31622565401011e-15, 6.932658445981469e-15]
u_inftys_sigmoidELM = [0.10275754512016544, 2.6305830356010418e-05, 2.481517692165536e-07, 5.724550167229836e-09]
u_rel_l2_sigmoidELM = [0.007353020014949969, 2.487283212831957e-06, 1.8385980714344105e-08, 3.852355093302462e-10]
u_inftys_swishELM = [0.06304677763961042, 1.823901803632566e-05, 1.1645443986907367e-07, 1.1023086443628927e-08]
u_rel_l2_swishELM = [0.004554774480607724, 1.568399218835126e-06, 6.2690229240207745e-09, 9.179413591137778e-10]
u_inftys_tanhELM = [0.14518350075984499, 1.9460227032475075e-05, 2.441407649378391e-07, 1.045405184640913e-08]
u_rel_l2_tanhELM = [0.007246766853804078, 1.1581280136845742e-06, 1.735290800390618e-08, 4.3532634203162415e-10]

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

root_path = '/home/lunxu/yangqh/FEN/Plot_Fig/Linear/Diffusion'
file_name =  root_path + '/error_Diffusion'
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