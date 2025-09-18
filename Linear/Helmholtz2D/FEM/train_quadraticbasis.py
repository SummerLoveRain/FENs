import sys
from model import *
from init_config import *
from train_config import *
import scipy.io as sio
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from utils import *
from integration import nodes_weights_2D_dict
from tqdm import tqdm
from scipy.interpolate import griddata
from plot.heatmap import plot_heatmap3

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=16)

class Boundary:
    DIRICHLET_BOUNDARY = 0
    NEUMANN_BOUNDARY = 1
    ROBIN_BOUNDARY = 2
    
if __name__ == "__main__":
    # 设置需要写日志
    init_log()
    # cuda 调用
    device = get_device(sys.argv)

    # 给定三角元的六个点
    X = [[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]]
    X = np.asarray(X)
    # 给定神经网络模型, 输出为六个基函数
    model_name = MODEL_NAME.Quadratic2DBasis
    model_dict = {
        'init_type': INIT_TYPE.Kaiming_Normal,
    }
    model = get_model(model_name=model_name, model_dict=model_dict)
    model.to(device)

    # 从mat文件中读取网格信息
    mesh_mat = sio.loadmat(getParentDir() + '/' + getCurDirName() + '/mesh0005.mat')    
    P = mesh_mat['P']
    T = np.asarray(mesh_mat['T'], dtype=int)
    Pb = mesh_mat['Pb']
    Tb = np.asarray(mesh_mat['Tb'], dtype=int)
    edges = np.asarray(mesh_mat['edges'], dtype=int)
    # 单元数量
    Ne = T.shape[0]
    # 点数量
    Np = Pb.shape[0]
    log("Np: %d" % Np)
    # 基函数个数
    Nb = 6
    A = lil_matrix((Np, Np))
    b = lil_matrix((Np, 1))
    # 定义积分点数目
    N_Integral = 10
    # 获取积分点和权重
    nodes, weights = nodes_weights_2D_dict[N_Integral]
    x = nodes[:, 0:1]
    y = nodes[:, 1:2]
    x = data_loader(x, device=device)
    y = data_loader(y, device=device)
    weights = np.transpose(weights, axes=(1, 0))
    # weights = np.reshape(weights, (weights.shape[0], 1, 1))
    weights = data_loader(weights, device=device)
    
    log('遍历计算单元')
    # 遍历每个单元
    for i in tqdm(range(Ne)):  # 基函数涉及边界单元,对应右边
        # 获取三角元的三个点
        vertices = P[T[i]]
        vertices = data_loader(vertices, device=device)
        # 将积分点根据三角元的坐标进行坐标变换
        new_x, new_y, J = triangluar_shift(x=x, y=y, vertices=vertices)
        # 将坐标移动到正规三角形中
        new_x1, new_y1 = affine_triangular_xy(x=new_x, y=new_y, vertices=vertices)
        newX = torch.cat([new_x1, new_y1], dim=1)
        us = model(newX)
        us_x = []
        us_y = []
        for j in range(Nb):
            u_x = compute_grad(us[:, j:j+1], new_x)
            u_x = reload(u_x, device=device)
            u_y = compute_grad(us[:, j:j+1], new_y)
            u_y = reload(u_y, device=device)
            us_x.append(u_x)
            us_y.append(u_y)
        us_x = torch.cat(us_x, dim=1)
        us_y = torch.cat(us_y, dim=1)
        # 计算积分
        us_x_vs_x = torch.einsum("ki, kj->kij", us_x, us_x)
        us_y_vs_y = torch.einsum("ki, kj->kij", us_y, us_y)
        us_vs = torch.einsum("ki, kj->kij", us, us)
        terms = -(us_x_vs_x + us_y_vs_y) + k**2*us_vs
        w = torch.unsqueeze(weights, dim=2)
        integrals = w*terms
        integrals = 0.5*torch.sum(integrals, dim=0)*J
        integrals = detach(integrals)

        for j in range(Nb):
            for m in range(Nb):
                A[Tb[i, j], Tb[i, m]] += integrals[j, m]

        # 处理右端项
        q = funQ(a1, a2, k, new_x, new_y, type='torch')
        qvs = q*us
        integrals = weights*qvs
        integrals = 0.5*torch.sum(integrals, dim=0)*J
        integrals = detach(integrals)
        for j in range(Nb):
            b[Tb[i, j], 0] += integrals[j]

    # 处理边界条件
    for edge in edges:
        if edge[0] == Boundary.DIRICHLET_BOUNDARY:
            for j in range(2, 5):
                index = edge[j]
                x = Pb[index, 0]
                y = Pb[index, 1]
                A[index] = 0
                A[index, index] = 1
                h = funH(a1, a2, k, x, y, type='numpy')
                b[index, 0] = h

    # 求解线性方程组
    A = A.tocsc()
    b = b.tocsc()
    uh = spsolve(A, b)

    uh = np.reshape(uh, (Np, 1))
    loss_A = np.max(np.abs(A.dot(uh) - b))
    log('loss_A: %.4E' % loss_A)

    u_true = funU(a1, a2, k, Pb[:, 0:1], Pb[:, 1:2], type='numpy')
    u_pred = uh
    
    u_infty = np.max(np.linalg.norm(u_true-u_pred, ord=np.inf))
    u_rel_l2 = np.max(np.linalg.norm(u_true-u_pred)/np.linalg.norm(u_true))
    log_str = 'u_infty %.4E u_rel_l2 %.4E' % (u_infty, u_rel_l2)
    log(log_str)

    # 均匀网格点绘图    
    GRID_SIZE = 100
    X = np.linspace(lb[0], ub[0], GRID_SIZE+1)
    Y = np.linspace(lb[1], ub[1], GRID_SIZE+1)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X_TRAIN, Y_TRAIN = np.meshgrid(X, Y)
    
    X_Train = np.hstack((X_TRAIN.flatten()[:, None], Y_TRAIN.flatten()[:, None]))
    x = X_Train[:, 0:1]
    y = X_Train[:, 1:2]
    u_true = funU(a1, a2, k, x, y, type='numpy')

    def solve(x, y):
        with torch.no_grad():
            x = data_loader(x, device=device)
            y = data_loader(y, device=device)

            vertices = P[T]
            vertices = data_loader(vertices, device=device)
            isIn = is_in_triangle(x, y, vertices)
            isIn = torch.transpose(isIn, 1, 0)
            coe = torch.sum(isIn, dim=1, keepdim=True)

            u = 0*x
            for i in range(Ne):  # 基函数涉及边界单元,对应右边
                vertices = P[T[i]]
                vertices = data_loader(vertices, device=device)
                new_x, new_y = affine_triangular_xy(x=x, y=y, vertices=vertices)
                X = torch.cat([new_x, new_y], dim=1)
                us = model(X)
                isIn = is_in_triangle(x, y, vertices)
                isIn = torch.unsqueeze(isIn, dim=1)
                us = us*isIn
                isIn = detach(isIn)
                for j in range(Nb):
                    u = us[:, j:j+1]*uh[Tb[i, j], 0] + u
            u = u/coe
        return u

    
    # def solve(x, y):
    #     with torch.no_grad():
    #         x = data_loader(x, device=device)
    #         y = data_loader(y, device=device)

    #         vertices = P[T]
    #         vertices = data_loader(vertices, device=device)
    #         isIn = is_in_triangle(x, y, vertices)
    #         isIn = torch.transpose(isIn, 1, 0)
    #         coe = torch.sum(isIn, dim=1, keepdim=True)

    #         new_x, new_y = affine_triangular_xy(x=x, y=y, vertices=vertices)
    #         new_x1 = torch.reshape(new_x, (new_x.shape[0]*new_x.shape[1], 1))
    #         new_y1 = torch.reshape(new_y, (new_y.shape[0]*new_y.shape[1], 1))
    #         X = torch.cat([new_x1, new_y1], dim=1)
    #         us = model(X)
    #         us = torch.reshape(us, (new_y.shape[0], new_x.shape[1], Nb))
    #         us = torch.transpose(us, 1, 0)
    #         w = uh[Tb]
    #         w = np.reshape(w, (1, Ne, Nb))
    #         w = data_loader(w)
    #         # u = torch.matmul(us, w)
    #         us = us*w
    #         us = torch.sum(us, dim=2)
    #         us = us*isIn
    #         u = torch.sum(us, dim=1, keepdim=True)
    #         u = u/coe
    #     return u

    n_batch = 1000
    n_all = x.shape[0]
    n_batchs = n_all//n_batch
    if n_all%n_batch != 0:
        n_batchs += 1
    
    u_pred = []
    for i in tqdm(range(n_batchs)):
        xt = x[i*n_batch:(i+1)*n_batch, :]
        yt = y[i*n_batch:(i+1)*n_batch, :]
        u_predt = solve(xt, yt)
        u_pred.append(u_predt)
    u_pred = torch.cat(u_pred, dim=0)
    u_pred = detach(u_pred)
    
    u_infty = np.max(np.linalg.norm(u_true-u_pred, ord=np.inf))
    u_rel_l2 = np.max(np.linalg.norm(u_true-u_pred)/np.linalg.norm(u_true))
    log_str = 'u_infty %.4E u_rel_l2 %.4E' % (u_infty, u_rel_l2)
    log(log_str)

    
    U_star = griddata(X_Train, u_true.flatten(), (X_TRAIN, Y_TRAIN), method='cubic')
    U_pred = griddata(X_Train, u_pred.flatten(), (X_TRAIN, Y_TRAIN), method='cubic')
    file_name = log_path + '/heatmap3'
    # plot_heatmap3(X, Y, T, P, E=None, xlabel=None, ylabel=None, T_title=None, P_title=None, E_title=None, file_name=None, abs=True):
    plot_heatmap3(X=X_TRAIN, Y=Y_TRAIN, T=U_star, P=U_pred, E=None, xlabel='x',
                ylabel='y', file_name=file_name)