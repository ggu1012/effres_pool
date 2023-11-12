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

def pklz_to_incmat(pkl):
    """
    Convert obtained dataset (.pklz) to hypergraph incidence matrix (scipy.sparse.csc)
    - Remove singleton edges and nodes
    return: H (scipy.sparse), net2node (list), net_mapper (list)
    """

    pin2net = np.array(pkl["pin_info"]["pin2net_map"])
    pin2node = np.array(pkl["pin_info"]["pin2node_map"])
    pin_direction = np.array(pkl["pin_info"]["pin_direct"])
    net_num = max(pin2net) + 1

    # init hypergraph incidence matrix
    H = csc_array((np.ones(len(pin2net)), (pin2node, pin2net)))
    # find and remove singleton edge
    e_deg = H.sum(axis=0).flatten()
    singleton_edge_idx = np.nonzero(e_deg <= 1)[0]

    net2node = [[] for _ in range(H.shape[1])]
    net_out = -1 * np.ones(net_num, dtype=int)

    # find and remove net with no output pin
    for pin in range(len(pin2node)):
        v = pin2node[pin]
        e = pin2net[pin]
        dir = pin_direction[pin]
        net2node[e].append(v)
        if dir == b"OUTPUT":
            net_out[e] = v
    no_out_net_idx = np.nonzero(net_out == -1)[0]

    net2node = np.array([list(set(x)) for x in net2node], dtype=object)

    ### incidence matrix w/ sparse mat.
    remove_net_idx = np.union1d(no_out_net_idx, singleton_edge_idx)

    # remove singleton hedge in inc mat
    if len(remove_net_idx) > 0:
        new_net_mapper = np.delete(np.arange(H.shape[1]), remove_net_idx)
    else:
        new_net_mapper = np.arange(H.shape[1])
    H = H[:, np.array(new_net_mapper)]

    void_node = np.nonzero(H.sum(axis=1) == 0)[0]
    new_node_mapper = np.delete(np.arange(H.shape[0]), void_node)
    H = H[new_node_mapper]

    indptr = H.indptr
    indices = H.indices

    ### net2node generation
    # remove void node
    net2node = []
    for i in range(len(indptr) - 1):
        net = indptr[i]
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
    
