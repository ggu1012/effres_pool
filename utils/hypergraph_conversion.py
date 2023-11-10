import numpy as np
from scipy.sparse import csr_array, csc_array, diags
from math import ceil, floor
import numpy as np
import dgl
import torch
from scipy.sparse import csc_array

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
    hinc: list, W: list, expander_sz: int
):  # expander_sz=1 for 3-edge cycles
    src = []
    dst = []
    wts = []
    rng = np.random.default_rng(462346234)

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
    # A = csr_array((wts, (src, dst)), shape=(num_nodes + len(hinc), num_nodes + len(hinc)))

    return A


def driver2load(obj):
    num_nodes = (
        obj["db_place_info"]["num_movable_nodes"]
        + obj["db_place_info"]["num_terminals"]
        + obj["db_place_info"]["num_terminal_NIs"]
    )

    pin2node = obj["pin_info"]["pin2node_map"]
    pin2net = obj["pin_info"]["pin2net_map"]
    pin_direction = obj["pin_info"]["pin_direct"]

    net_num = max(pin2net) + 1
    net2node = [[] for _ in range(net_num)]
    net_out = -1 * np.ones(net_num, dtype=int)

    for pin in range(len(pin2node)):
        v = pin2node[pin]
        e = pin2net[pin]
        dir = pin_direction[pin]
        net2node[e].append(v)
        if dir == b"OUTPUT":
            net_out[e] = v
    no_out_net = np.nonzero(net_out == -1)[0]
    net_mapid = np.delete(np.arange(net_num), no_out_net)
    net2node = np.array([list(set(x)) for x in net2node], dtype=object)

    rev_netmap = {}
    for i, x in enumerate(net_mapid):
        rev_netmap[x] = i

    net2node = np.delete(net2node, no_out_net).tolist()
    net_out = np.delete(net_out, no_out_net)

    row = []
    col = []
    src_dst = {}
    for i, net in enumerate(net2node):
        dsts = np.delete(net, np.nonzero(net == net_out[i])[0])
        src_dst[net_mapid[i]] = (net_out[i], dsts)
        row += [net_out[i] for _ in range(len(net) - 1)]
        col += dsts.tolist()

    A = csr_array((np.ones(len(row), dtype=int), (row, col)), (num_nodes, num_nodes))
    return A + A.T, src_dst, no_out_net

def star_hetero(H):
    
    ##          (net-to-node)
    ##    [ 0       H.T
    ##      H        0 ]
    ## (node-to-net)

    A_node2net = H
    A_net2node = H.T

    return dgl.heterograph(
        {
            ('node', 'to', 'net'): ('csc', (torch.tensor(A_node2net.indptr, dtype=int), torch.tensor(A_node2net.indices, dtype=int), [])), 
            ('net', 'to', 'node'): ('csr', (torch.tensor(A_net2node.indptr, dtype=int), torch.tensor(A_net2node.indices, dtype=int), []))
        }
    )



