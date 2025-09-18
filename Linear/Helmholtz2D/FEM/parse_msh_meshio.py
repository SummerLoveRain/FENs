import pygmsh
import meshio
from init_config import init_log, log_path, getParentDir, getCurDirName
import numpy as np
import scipy.io
from tqdm import tqdm

# 边界
edges=[]

class BOUNDARY_TYPE:
    DIRICHLET = 'Dirichlet'
    NEUMANN = 'Neumann'
    ROBIN = 'Robin'

class Boundary:
    DIRICHLET_BOUNDARY = 0
    NEUMANN_BOUNDARY = 1
    ROBIN_BOUNDARY = 2

mesh = meshio.read(getParentDir() + '/' + getCurDirName() + '/mesh.msh')

P = mesh.points[:, :2]
T = mesh.cells_dict['triangle'][:, :3]
lines = mesh.cells_dict['line'][:, :2]

# 0:边界类型，1:所属单元，2:起始点，3:终止点
edges = np.zeros((len(lines), 5), dtype=int)
edges[:, 2:4] = lines

# for i in mesh.cell_sets_dict[BOUNDARY_TYPE.NEUMANN]['line']:
#     edges[i, 0] = Boundary.NEUMANN_BOUNDARY
# for i in mesh.cell_sets_dict[BOUNDARY_TYPE.ROBIN]['line']:
#     edges[i, 0] = Boundary.ROBIN_BOUNDARY
for i in mesh.cell_sets_dict[BOUNDARY_TYPE.DIRICHLET]['line']:
    edges[i, 0] = Boundary.DIRICHLET_BOUNDARY

# 遍历所有单元，记录所有线段所在单元
print("遍历所有单元，记录所有线段所在单元")
lines_dict = {}
for i in tqdm(range(len(T))):
    for j in range(3):
        a = T[i, j]
        b = T[i, (j+1)%3]
        if a>b:
            a,b = b,a
        if (a,b) in lines_dict:
            lines_dict[(a,b)].append(i)
        else:
            lines_dict[(a,b)] = [i]

print("判断边界所属单元，并添加到edges矩阵中")
for i in tqdm(range(len(edges))):
    a = edges[i, 2]
    b = edges[i, 3]
    if a>b:
        a,b = b,a
    try:
        edges[i, 1] = lines_dict[(a,b)][0]
    except:
        print("Error: %s-%s not found in lines_dict" % (a,b))

# # 判断边界所属单元，并添加到edges矩阵中
# for i in range(len(edges)):
#     a = edges[i, 2]
#     b = edges[i, 3]
#     for j in range(len(T)):
#         if a in T[j] and b in T[j]:
#             edges[i, 1] = j


# 在三角形的每条边上增加一个点
def edge_idx2str(a, b):
    if a<b:
        return str(a) + '-' + str(b)
    else:
        return str(b) + '-' + str(a)

N_Element = T.shape[0]

Pb = P
Tb = np.zeros((N_Element, 6))
Tb[:, :3] = T
midp_dict = {}
print("处理三角元")
for elid, element in enumerate(tqdm(mesh.cells_dict['triangle'])):
    # print("Element id = %s, connectivity = %s" % (elid, element))
    data = element
    a = data[0]
    b = data[1]
    c = data[2]

    P_a = Pb[a]
    P_b = Pb[b]
    P_ab = (Pb[a] + Pb[b])/2
    m_ab = len(Pb)
    ab = edge_idx2str(a, b)
    try:
        m_ab = midp_dict[ab]
        Tb[elid, 3] = m_ab
    except:
        midp_dict[ab] = m_ab
        Tb[elid, 3] = m_ab
        Pb = np.concatenate((Pb, [P_ab]), axis=0)

    P_bc = (Pb[b] + Pb[c])/2
    m_bc = len(Pb)
    bc = edge_idx2str(b, c)
    try:
        m_bc = midp_dict[bc]
        Tb[elid, 4] = m_bc
    except:
        midp_dict[bc] = m_bc
        Tb[elid, 4] = m_bc
        Pb = np.concatenate((Pb, [P_bc]), axis=0)

    P_ca = (Pb[c] + Pb[a])/2
    m_ca = len(Pb)
    ca = edge_idx2str(c, a)
    try:
        m_ca = midp_dict[ca]
        Tb[elid, 5] = m_ca
    except:
        midp_dict[ca] = m_ca
        Tb[elid, 5] = m_ca
        Pb = np.concatenate((Pb, [P_ca]), axis=0)



# 保存P,T, Pb,Tb矩阵到mat文件中
Pb = np.asarray(Pb)
P = np.asarray(P)
T = np.asarray(T)


print("添加中间点索引到edges矩阵中")
# 添加中间点索引到edges矩阵中
for i in tqdm(range(len(edges))):
    a = edges[i, 2]
    b = edges[i, 3]
    # 到所属单元中查询中间点
    elid = edges[i, 1]
    m = (Pb[a]+Pb[b])/2
    idxs = np.asarray(Tb[elid], dtype=int)
    ps = Pb[idxs]
    idx = 0
    for p in ps:
        if (p==m).all():
            edges[i, 4] = idxs[idx]
            break
        else:
            idx += 1

data = {'P':P, 'T':T, 'Pb':Pb, 'Tb':Tb, 'edges':edges}
scipy.io.savemat(getParentDir() + '/' + getCurDirName() + '/mesh.mat', data)

# 画图
meshio.svg.write(getParentDir() + '/' + getCurDirName() + '/mesh.svg', mesh)