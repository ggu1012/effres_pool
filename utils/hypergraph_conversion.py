import numpy as np
from scipy.sparse import csr_array, csc_array, diags
from math import ceil, floor
import numpy as np
import dgl
import torch
from scipy.sparse import csc_array
import dgl.sparse as dglsp
from typing import List
from utils.functions import lil_to_dglsp

def StarW(hinc: list, W: list):
    num_nodes = 0
    for x in hinc:
        for y in x:
            if y > num_nodes:
                num_nodes = y
    num_nodes += 1

    sz = len(hinc)
    row2 = []
    col2 = []
    val2 = []

    ## [ 0 H.T
    ##   H  0 ]
    for eid in range(len(hinc)):
        esz = len(hinc[eid])
        cc2 = [eid + num_nodes for _ in range(esz)]
        vv2 = [W[eid] / esz for _ in range(esz)]
        rr2 = hinc[eid]

        row2 += rr2
        col2 += cc2
        val2 += vv2

    mat2 = csr_array((val2, (row2, col2)), shape=(sz + num_nodes, sz + num_nodes))

    return mat2 + mat2.T


def FullCliqueW(hinc: list, W: list):
    enum = len(hinc)
    colidx = []
    colptr = []
    colptrx = 0
    ewts = []
    ## H @ H.T
    for n in range(enum):
        esz = len(hinc[n])
        colptr.append(colptrx)
        colidx += hinc[n].tolist()
        colptrx += esz
        ewts.append(W[n] / esz)
    colptr.append(colptrx)
    De_inv = diags(ewts)
    H = csc_array((np.ones(len(colidx)), colidx, colptr))
    clique = H @ De_inv @ H.T

    return clique


def ExpanderCliqueW(
    hinc: list, expander_sz: int, random_seed: int = 54321, W: list | int = -1
) -> csr_array:  # expander_sz=1 for 3-edge cycles
    
    if W == -1:
        W = np.ones(len(hinc))

    src = []
    dst = []
    wts = []
    rng = np.random.default_rng(random_seed)

    for eid, hedge in enumerate(hinc):
        hsz = len(hedge)
        if hsz == 1:
            src.append(hedge[0])
            dst.append(hedge[0])
            wts.append(W[eid])
        elif hsz == 2:
            src.append(hedge[0])
            dst.append(hedge[1])
            wts.append(W[eid])
        elif hsz == 3:
            src.append(hedge[0])
            dst.append(hedge[1])

            src.append(hedge[1])
            dst.append(hedge[2])

            src.append(hedge[2])
            dst.append(hedge[0])
            wts += [W[eid] / (hsz - 1) for _ in range(hsz)]
        else:
            he = np.array(hedge, dtype=int)
            bw = floor(hsz / 2) * ceil(hsz / 2) / (hsz - 1)
            bw = 1 / (expander_sz * 2 * bw)

            for t in range(expander_sz):
                rng.shuffle(he)
                src += he[:-2].tolist()
                dst += he[1:-1].tolist()
                src.append(he[-1])
                dst.append(he[0])
                wts += [W[eid] * bw for _ in range(hsz - 1)]

    num_nodes = 0
    for x in hinc:
        for y in x:
            if y > num_nodes:
                num_nodes = y
    num_nodes += 1

    A = csr_array((wts, (src, dst)), shape=(num_nodes, num_nodes))
    A = A + A.T
    return A


def driver2load(H, obj, new_node_mapper, new_net_mapper):
    num_nodes, num_nets = H.shape
    net2node = [[] for _ in range(num_nets)]

    net_out = -1 * np.ones(num_nets, dtype=int)
    pin2net = obj["pin_info"]["pin2net_map"]
    pin2node = obj["pin_info"]["pin2node_map"]
    pin_direction = obj['pin_info']['pin_direct']

    rev_net_mapper = {}
    for i, nd in enumerate(new_net_mapper):
        rev_net_mapper[nd] = i
    rev_node_mapper = {}
    for i, nd in enumerate(new_node_mapper):
        rev_node_mapper[nd] = i

    for pin in range(len(pin2node)):
        v = pin2node[pin]
        e = pin2net[pin]
        if v not in rev_node_mapper.keys() \
            or e not in rev_net_mapper.keys():
            continue
        dir = pin_direction[pin]
        new_e = rev_net_mapper[e]
        new_v = rev_node_mapper[v]
        net2node[new_e].append(new_v)
        if dir == b"OUTPUT":
            net_out[new_e] = new_v

    src_dst = {}

    row = []
    col = []
    for i, net in enumerate(net2node):
        dsts = np.delete(net, np.nonzero(net == net_out[i])[0])
        src_dst[i] = (net_out[i], dsts.tolist())
        row += [net_out[i] for _ in range(len(dsts))]
        col += dsts.tolist()
    A = csr_array((np.ones(len(row), dtype=int), (row, col)), (num_nodes, num_nodes))
    return A + A.T, src_dst


def star_hetero(H):
    ##          (net-to-node)
    ##    [ 0       H.T
    ##      H        0 ]
    ## (node-to-net)

    A_node2net = H
    A_net2node = H.T

    return dgl.heterograph(
        {
            ("node", "to", "net"): (
                "csc",
                (
                    torch.tensor(A_node2net.indptr, dtype=int),
                    torch.tensor(A_node2net.indices, dtype=int),
                    [],
                ),
            ),
            ("net", "to", "node"): (
                "csr",
                (
                    torch.tensor(A_net2node.indptr, dtype=int),
                    torch.tensor(A_net2node.indices, dtype=int),
                    [],
                ),
            ),
        }
    )

def multi_level_expander_graph(net2nodes: list, assignment_mats: List[dglsp.SparseMatrix], expander_sz=3, device='cpu'):

    ## Build uniform 3-cycle expander graph from net2node list (incidence matrix)

    assert len(net2nodes) == len(assignment_mats) + 1

    H = lil_to_dglsp(net2nodes[0])

    data_dict = {}
    for lv, n2n in enumerate(net2nodes):
        key = (f'lv{lv}', 'to', f'lv{lv}')
        adj = ExpanderCliqueW(hinc=n2n, expander_sz=expander_sz)
        indptr = torch.tensor(adj.indptr, dtype=int)
        indices = torch.tensor(adj.indices, dtype=int)
        data_dict[key] = ('csr', (indptr, indices, []))
    
    for lv in range(len(net2nodes) - 1):
        data_dict[(f'lv{lv}', 'downwards', f'lv{lv+1}')] = \
                ('csr', (assignment_mats[lv].T.csr()))
        data_dict[(f'lv{lv+1}', 'upwards', f'lv{lv}')] = \
                ('csr', (assignment_mats[lv].csr()))
    
    data_dict[('lv0', 'conn', 'net')] = ('csr', H.csr())
    gr = dgl.heterograph(data_dict).to(device)
    
    return gr