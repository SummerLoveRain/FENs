u_inftys_cosELM = [1.836160662677001e-07, 1.272721241643445e-14, 9.648139706982676e-15, 3.315403507286874e-14]
u_rel_l2_cosELM = [7.446549097012005e-08, 2.3325477671841425e-15, 1.6508455847525744e-15, 3.55493872633844e-15]
u_inftys_csELM = [2.1902538100554425e-07, 1.960132081072728e-14, 6.4755727624456226e-15, 2.6069424396979457e-14]
u_rel_l2_csELM = [7.857594608027608e-08, 4.668303338605767e-15, 1.4677128008994636e-15, 2.3769781435411715e-15]
u_inftys_sinELM = [2.2316089610491534e-07, 1.0183596099121764e-14, 1.509903313490213e-14, 1.7819079545233762e-14]
u_rel_l2_sinELM = [9.556708274587606e-08, 3.947475475109592e-15, 2.011393912748518e-15, 3.4593299814416995e-15]
u_inftys_sigmoidELM = [3.242492675810043e-05, 6.379559636116028e-08, 4.0454324948815624e-09, 1.7598722479306161e-10]
u_rel_l2_sigmoidELM = [9.220885540456844e-06, 6.261106036582125e-09, 2.771374477655902e-10, 1.801271596208364e-11]
u_inftys_swishELM = [3.385543823239142e-05, 3.585591912269592e-08, 9.677023934448792e-10, 3.78349887727909e-10]
u_rel_l2_swishELM = [1.093977130498558e-05, 4.531784834066339e-09, 1.689207947545941e-10, 2.0718179299100182e-11]
u_inftys_tanhELM = [2.741813659664923e-05, 1.2665987014770508e-07, 3.1868693736167633e-09, 1.978150976356119e-10]
u_rel_l2_tanhELM = [6.137036869700112e-06, 1.078011772069569e-08, 2.4560402115276503e-10, 1.8774987669446285e-11]

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

root_path = '/home/lunxu/yangqh/FEN/Plot_Fig/Func2D'
file_name =  root_path + '/error_Func2D'
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