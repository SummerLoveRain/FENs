from init_config import *
import gmshparser
import numpy as np
import scipy.io

mesh = gmshparser.parse(getParentDir() + '/' + getCurDirName() + '/mesh.msh')

PT = True

P = []
T = []

Pb = []
Tb = []

# 边界
edges=[]
class Boundary:
    DIRICHLET_BOUNDARY = 0
    NEUMANN_BOUNDARY = 1
    ROBIN_BOUNDARY = 2

idx = 0
for entity in mesh.get_node_entities():
    for node in entity.get_nodes():
        nid = node.get_tag()
        ncoords = node.get_coordinates()
        print("Node id = %s, node coordinates = %s" % (nid, ncoords))
        if PT:
            idx = idx+1
            data = ncoords[0:2]
            data = np.asarray(data)
            P.append(data)
            Pb.append(data)

idx = 0
count_type1 = 0
entity_idx=0
for entity in mesh.get_element_entities():
    eltype = entity.get_element_type()
    print("Element type: %s" % eltype)
    entity_idx = entity_idx + 1
    for element in entity.get_elements():
        elid = element.get_tag()
        elcon = element.get_connectivity()
        print("Element id = %s, connectivity = %s" % (elid, elcon))
        if eltype == 2 and PT:
            tdata = np.asarray(elcon[:])-1
            T.append(tdata)
            data = np.zeros((1, 6))
            data[:, 0:3] = tdata[:]
            data = np.asarray(data)
            Tb.append(data)
            idx = idx+1
        else:
            count_type1 = count_type1 + 1
            boundary_type = None
            if entity_idx==1:
                n = [0, -1]
                boundary_type = Boundary.NEUMANN_BOUNDARY
            elif entity_idx==2:
                n = [1, 0]
                boundary_type = Boundary.ROBIN_BOUNDARY
            elif entity_idx==3:
                n = [0, 1]
                boundary_type = Boundary.ROBIN_BOUNDARY
            elif entity_idx==4:
                n = [1, 0]
                boundary_type = Boundary.ROBIN_BOUNDARY
            elif entity_idx==5:
                n = [0, 1]
                boundary_type = Boundary.ROBIN_BOUNDARY
            elif entity_idx==6:
                n = [1, 0]
                boundary_type = Boundary.ROBIN_BOUNDARY
            elif entity_idx==7:
                n = [-1, 1]
                boundary_type = Boundary.NEUMANN_BOUNDARY
            data = np.zeros((1, 5))
            data[:, 0:1] = boundary_type
            data[:, 1:3] = np.asarray(elcon[:])-1
            data[:, 3:5] = n[:]
            data = np.asarray(data)
            edges.append(data)
            

# 在三角形的每条边上增加一个点
def edge_idx2str(a, b):
    if a<b:
        return str(a) + '-' + str(b)
    else:
        return str(b) + '-' + str(a)

Tb = np.asarray(Tb)
Tb = Tb.squeeze()
edges = np.asarray(edges)
edges = edges.squeeze()

midp_dict = {}
for entity in mesh.get_element_entities():
    eltype = entity.get_element_type()
    print("Element type: %s" % eltype)
    for element in entity.get_elements():
        elid = element.get_tag()
        elcon = element.get_connectivity()
        print("Element id = %s, connectivity = %s" % (elid, elcon))
        if eltype == 2 and PT:
            data = elcon[:]
            a = data[0]-1
            b = data[1]-1
            c = data[2]-1

            idx = elid - count_type1 - 1

            P_a = Pb[a]
            P_b = Pb[b]
            P_ab = (Pb[a] + Pb[b])/2
            m_ab = len(Pb)
            ab = edge_idx2str(a, b)
            try:
                m_ab = midp_dict[ab]
                Tb[idx, 3] = m_ab
            except:
                midp_dict[ab] = m_ab
                Tb[idx, 3] = m_ab
                Pb.append(P_ab)

            P_bc = (Pb[b] + Pb[c])/2
            m_bc = len(Pb)
            bc = edge_idx2str(b, c)
            try:
                m_bc = midp_dict[bc]
                Tb[idx, 4] = m_bc
            except:
                midp_dict[bc] = m_bc
                Tb[idx, 4] = m_bc
                Pb.append(P_bc)
    
            P_ca = (Pb[c] + Pb[a])/2
            m_ca = len(Pb)
            ca = edge_idx2str(c, a)
            try:
                m_ca = midp_dict[ca]
                Tb[idx, 5] = m_ca
            except:
                midp_dict[ca] = m_ca
                Tb[idx, 5] = m_ca
                Pb.append(P_ca)


if PT:
    Pb = np.asarray(Pb)
    P = np.asarray(P)
    T = np.asarray(T)
    data = {'P':P, 'T':T, 'Pb':Pb, 'Tb':Tb, 'edges':edges}
    scipy.io.savemat(getParentDir() + '/' + getCurDirName() + '/mesh.mat', data)

# 画图
X, Y, T = gmshparser.helpers.get_triangles(mesh)

import matplotlib.pylab as plt
plt.figure()
plt.triplot(X, Y, T, color='black')
plt.axis('equal')
plt.axis('off')
plt.tight_layout()

if not PT:
    for entity in mesh.get_node_entities():
        for node in entity.get_nodes():
            nid = node.get_tag()
            ncoords = node.get_coordinates()
            # print("Node id = %s, node coordinates = %s" % (nid, ncoords))
            x = ncoords[0]
            y = ncoords[1]
            plt.text(x, y, str(nid), fontsize=5, color='red')
else:
    idx = 0
    for node in P:
        idx = idx + 1
        nid = idx
        x = node[0]
        y = node[1]
        plt.text(x, y, str(nid), fontsize=5, color='red')



plt.savefig(getParentDir() + '/' + getCurDirName() + '/mesh.svg')