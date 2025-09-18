u_inftys_cosELM = [1.8058643469487379e-09, 8.993540719481885e-06, 0.00036968035293072354, 0.004377087043679395]
# u_rel_l2_cosELM = [3.653032054307503e-10, 3.8060169128180174e-06, 9.095659050612269e-05, 0.004095797475724249]
u_inftys_FourierNet = [1.3934871034848584e-09, 2.337401985158527e-06, 0.00014656014601183154, 0.0009262479629022213]
# u_rel_l2_FourierNet = [3.0936125861842147e-10, 1.0415653562401313e-06, 8.127930699421621e-05, 0.0006223257127396287]
u_inftys_sinELM = [5.507039851915252e-09, 4.532622389197272e-06, 0.0003611553587445915, 0.0002855022555057207]
# u_rel_l2_sinELM = [1.0848023500264947e-09, 1.378334424039513e-06, 0.0002145394173374378, 0.00022569514248903596]
u_inftys_sigmoidELM = [4.733705871728944e-09, 1.9765716234854125e-06, 0.00024107780053173045, 0.00028907853419224416]
# u_rel_l2_sigmoidELM = [1.108512033738634e-09, 5.39538255637548e-07, 0.0001688539581945905, 0.00022652466130909164]
u_inftys_swishELM = [7.440823512894212e-09, 2.0388504074930758e-05, 0.00040591601696049473, 0.0046748425779223335]
# u_rel_l2_swishELM = [1.1649651164942497e-09, 5.997379991518915e-06, 0.00010332635854249304, 0.0043651651865130175]
u_inftys_tanhELM = [3.969412964988095e-09, 1.93838080408959e-06, 0.00020425413566060868, 0.0002881248598758379]
# u_rel_l2_tanhELM = [9.952109566337974e-10, 5.614613558093691e-07, 0.00016263723355553131, 0.0002255831985037109]


import numpy as np
basis = [5, 7, 10, 15]

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