import numpy as np
from scipy.sparse import csr_array, csc_array, diags
from math import ceil, floor
import gzip
import numpy as np
import pickle
import dgl
import torch
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

    # init hypergraph incidence matrix
    H = csc_array((np.ones(len(pin2net)), (pin2node, pin2net)))

    # find and remove singleton edge
    e_deg = H.sum(axis=0).flatten()
    singleton_edge_idx = np.nonzero(e_deg <= 1)[0]

    net2node = [[] for _ in range(H.shape[1])]

    for pin, net in enumerate(pin2net):
        net2node[net].append(pin2node[pin])

    ### incidence matrix w/ sparse mat.
    # remove singleton hedge in inc mat
    new_net_mapper = np.delete(np.arange(H.shape[1]), singleton_edge_idx)
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