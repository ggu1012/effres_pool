import numpy as np
from scipy.sparse import csr_array, csc_array, diags
from math import ceil, floor
import gzip
import numpy as np
import pickle
import dgl
import torch
from torch.utils.data import Dataset
from scipy.sparse import csc_array
import dgl.sparse as dglsp
import os
import re

def pklz_to_incmat(dname, pkl, debug=False):
    """
    Convert obtained dataset (.pklz) to hypergraph incidence matrix (scipy.sparse.csc)
    - Remove singleton edges and nodes
    return: H (scipy.sparse), net2node (list), net_mapper (list)
    """

    basen = os.path.basename(dname)
    top_var = re.match(r'^(.*)\.icc2\.pklz$', basen).group(1)

    pin2net = np.array(pkl["pin_info"]["pin2net_map"])
    pin2node = np.array(pkl["pin_info"]["pin2node_map"])
    pin_direction = np.array(pkl["pin_info"]["pin_direct"])
    net_num = max(pin2net) + 1

    # init hypergraph incidence matrix
    H = csc_array((np.ones(len(pin2net)), (pin2node, pin2net)))
    net2node = [[] for _ in range(H.shape[1])]
    net_out = -1 * np.ones(net_num, dtype=int)

    ### REMOVE NET
    # 0. SINGLETON EDGE
    e_deg = H.sum(axis=0).flatten()
    singleton_edge_idx = set(np.nonzero(e_deg <= 1)[0].tolist())

    # 1. NO OUTPUT PIN
    for pin in range(len(pin2node)):
        v = pin2node[pin]
        e = pin2net[pin]
        dir = pin_direction[pin]
        net2node[e].append(v)
        if dir == b"OUTPUT":
            net_out[e] = v
    no_out_net_idx = set(np.nonzero(net_out == -1)[0].tolist())

    # 2. TOP 3 DEGREE NETS
    top3_deg = set(np.argsort(e_deg)[::-1][:3].tolist())

    # 3. NETS WITH DEGREE OVER 1000 
    over1000_deg = set(np.nonzero(e_deg > 1000)[0].tolist())

    # 4. clk / rst / clock / reset
    net_name2id_map = pkl['net_info']['net_name2id_map']
    clk_rst = []
    for x in net_name2id_map.keys():
        if re.match(r'.*(clk|rst|clock|reset).*', x):
            clk_rst.append(net_name2id_map[x])
    clk_rst = set(clk_rst)


    ### incidence matrix w/ sparse mat.
    remove_net_idx = np.array(list(singleton_edge_idx |clk_rst | over1000_deg \
                                   | top3_deg | no_out_net_idx))

    # remove nets in inc mat
    if len(remove_net_idx) > 0:
        new_net_mapper = np.delete(np.arange(H.shape[1]), remove_net_idx)
    else:
        new_net_mapper = np.arange(H.shape[1])
    H = H[:, np.array(new_net_mapper)]

    ### REMOVE NODES AFTER NET REMOVAL
    void_node = np.nonzero(H.sum(axis=1) == 0)[0]
    new_node_mapper = np.delete(np.arange(H.shape[0]), void_node)
    H = H[new_node_mapper]

    if debug:
        print("Removed net names ====")
        for x in pkl['net_info']['net_name2id_map'].keys():
            if pkl['net_info']['net_name2id_map'][x] in remove_net_idx:
                print(x)
        print("\n\nRemoved node names ====")
        for x in pkl['node_info']['node_name2id_map'].keys():
            if pkl['node_info']['node_name2id_map'][x] in void_node:
                print(x)
    indptr = H.indptr
    indices = H.indices

    ### net2node generation
    # remove void node
    net2node = []
    for i in range(len(indptr) - 1):
        nodes = indices[indptr[i]:indptr[i+1]]
        net2node.append(list(nodes))
    return H, net2node, new_net_mapper, new_node_mapper

def lil_to_dglsp(lil: list):
    colptrx = 0
    colptr = []
    colidx = []

    for nodes in lil:
        colptr.append(colptrx)
        colidx += nodes
        colptrx += len(nodes)

    colptr.append(colptrx)
    colptr = torch.tensor(colptr, dtype=int)
    colidx = torch.tensor(colidx, dtype=int)
    return dglsp.from_csc(colptr, colidx)

def coo_to_dglsp(row: np.array, col: np.array):
    _row = torch.tensor(row, dtype=int)
    _col = torch.tensor(col, dtype=int)
    return dglsp.spmatrix(torch.vstack([_row, _col]))

def xcollate_fn(data):
    xs, ys, graphs = map(list, zip(*data))
    graph_batch = dgl.batch(graphs)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0), graph_batch


class CustomDataset(Dataset):
    def __init__(self, xs, ys, graphs):
        self.xs = xs
        self.ys = ys
        self.graphs = graphs
        self.size = len(xs)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 반환값 : (a, b)
        # a : idx에 해당하는 인덱스만 1이고 나머지는 0인 크기 (1, 8)짜리 one-hot vector
        # b : idx
        return self.xs[idx], self.ys[idx], self.graphs[idx]
    
