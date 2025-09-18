u_inftys_cosELM = [3.739164355920366e-08, 2.1671553440683056e-13, 5.2735593669694936e-15, 2.4424906541753444e-15]
u_rel_l2_cosELM = [1.6164167849860815e-08, 1.271719946800651e-13, 2.3601278295533643e-15, 1.7658586498621899e-15]
u_inftys_csELM = [3.524149327915893e-08, 3.517186542012496e-13, 5.329070518200751e-15, 2.858824288409778e-15]
u_rel_l2_csELM = [1.0682362941860124e-08, 1.5920278181142149e-13, 2.001595758984418e-15, 2.6000402915423007e-15]
u_inftys_sinELM = [6.066574798069269e-08, 4.335351522222197e-13, 6.994405055138486e-15, 5.662137425588298e-15]
u_rel_l2_sinELM = [2.4111526782580204e-08, 2.1114509941681673e-13, 3.332677949561766e-15, 3.923181385039056e-15]
u_inftys_sigmoidELM = [2.443557605147362e-07, 1.9003637552472696e-10, 2.0691004465334117e-11, 2.1174173525650986e-12]
u_rel_l2_sigmoidELM = [6.459981603242989e-08, 1.3385879161275063e-10, 7.1024900671406504e-12, 1.0359275234566482e-12]
u_inftys_swishELM = [1.0241488529461051e-07, 8.549250196665525e-11, 9.094947017729282e-12, 1.4412221416293391e-12]
u_rel_l2_swishELM = [3.8084072256434396e-08, 3.371692625993457e-11, 4.1869395242923345e-12, 6.245124933458607e-13]
u_inftys_tanhELM = [1.7974525690078735e-07, 8.81390516127567e-11, 8.29913915367797e-12, 1.6200374375330284e-12]
u_rel_l2_tanhELM = [3.5758721604280865e-08, 4.382980960576975e-11, 3.599794267933071e-12, 5.175466723116827e-13]

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

root_path = '/home/lunxu/yangqh/FEN/Plot_Fig/Linear/WaveEquation'
file_name =  root_path + '/error_Wave'
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