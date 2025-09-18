u_inftys_cosELM = [1.2395640069939873e-13, 5.601018981948869e-09, 5.736037515746695e-06, 0.0003313999835927173]
u_rel_l2_cosELM = [3.9640579658202396e-14, 7.747260277232804e-10, 3.5995521794959944e-07, 5.945863487494155e-05]
u_inftys_csELM = [2.456368441983159e-13, 1.5341081738284856e-09, 2.9905770949634203e-06, 0.00025601458366714125]
u_rel_l2_csELM = [5.0407576978271594e-14, 3.2616475453696035e-10, 4.983599606081073e-07, 3.399754614271736e-05]
u_inftys_sinELM = [3.664291092775329e-13, 1.5148146070842472e-09, 1.1275173019686946e-06, 7.935613417786591e-05]
u_rel_l2_sinELM = [7.949346482526591e-14, 3.194284374157845e-10, 4.415933480600142e-07, 5.207027160750204e-05]
u_inftys_sigmoidELM = [1.8920753852569305e-11, 2.733723247771991e-09, 1.0510966591548154e-06, 6.237412509180196e-05]
u_rel_l2_sigmoidELM = [8.711642003021205e-12, 3.844999868752424e-10, 3.148684076228674e-07, 4.027929794971722e-05]
u_inftys_swishELM = [1.5993872892750005e-12, 5.223919741226268e-09, 5.5395827859294755e-06, 0.00033136550399487374]
u_rel_l2_swishELM = [4.5378253181791226e-13, 1.0120394878356846e-09, 4.276114977853498e-07, 5.939541857437993e-05]
u_inftys_tanhELM = [9.599668837623199e-10, 1.529113768938828e-08, 2.1064074005150424e-05, 0.000498231786634884]
u_rel_l2_tanhELM = [1.0015893047057113e-10, 1.9115193434840955e-09, 8.106270137550677e-06, 0.00031661596976433813]


import numpy as np
basis = [5, 7, 10, 15]

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

root_path = '/home/lunxu/yangqh/FEN/Plot_Fig/Linear/PoissonHD'
file_name =  root_path + '/error_PoissonHD'
xy_labels = ['d', 'error']
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