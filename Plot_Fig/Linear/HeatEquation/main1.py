u_inftys_cosELM = [1.5015166354714893e-07, 2.438493851286694e-12, 2.993161274389422e-13, 4.574118861455645e-14]
u_rel_l2_cosELM = [4.98629132316017e-08, 1.2761076994110967e-12, 1.288319498462934e-13, 2.890723167777328e-14]
u_inftys_csELM = [6.477203051691305e-08, 2.125299936039937e-12, 2.1938006966593093e-13, 3.182176744331855e-14]
u_rel_l2_csELM = [2.3951894547551195e-08, 9.136400146430718e-13, 1.3437426110318626e-13, 3.006305236106998e-14]
u_inftys_sinELM = [6.69928909569073e-08, 2.8705926524708048e-12, 1.2256862191861728e-13, 2.609024107869118e-14]
u_rel_l2_sinELM = [3.884675705592819e-08, 1.178627376419304e-12, 9.766272934134762e-14, 1.7562394342982924e-14]
u_inftys_sigmoidELM = [1.5832483768463135e-07, 2.473825588822365e-10, 3.001332515850663e-11, 5.993566754014523e-12]
u_rel_l2_sigmoidELM = [7.381743513723491e-08, 2.364278856734221e-10, 2.270511588550841e-11, 3.501500015631111e-12]
u_inftys_swishELM = [1.4388383928753967e-07, 1.4778218926370101e-10, 2.0668911027144077e-11, 2.4442670110147446e-12]
u_rel_l2_swishELM = [7.573094138862475e-08, 7.296008247141489e-11, 8.128724615505865e-12, 1.4761036668005425e-12]
u_inftys_tanhELM = [1.291045919060707e-07, 1.1386480647246344e-10, 1.5006662579253316e-11, 4.144240506320784e-12]
u_rel_l2_tanhELM = [5.336246488091466e-08, 5.731908297226275e-11, 7.76140416669787e-12, 3.0036952303088216e-12]


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

root_path = '/home/lunxu/yangqh/FEN/Plot_Fig/Linear/HeatEquation'
file_name =  root_path + '/error_Heat'
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