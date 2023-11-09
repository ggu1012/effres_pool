import numpy as np
from scipy.sparse import csr_array, csc_array, diags
from math import ceil, floor

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

    mat2 = csr_array((val2, (row2,col2)), shape=(sz + num_nodes, sz + num_nodes))

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
        ewts.append(W[n]/esz)
    colptr.append(colptrx)
    De_inv = diags(ewts)
    H = csc_array((np.ones(len(colidx)), colidx, colptr))
    clique = H @ De_inv @ H.T

    return clique


def ExpanderCliqueW(hinc: list, W: list, expander_sz: int): # expander_sz=1 for 3-edge cycles
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
            wts += [W[eid] / (hsz-1) for _ in range(hsz)]
        else:
            he = np.array(hedge, dtype=int)
            bw = floor(hsz/2) * ceil(hsz/2) / (hsz-1)
            bw = 1/(expander_sz * 2 * bw)

            for t in range(expander_sz):
                rng.shuffle(he)
                src += he[:-2].tolist()
                dst += he[1:-1].tolist()
                src.append(he[-1])
                dst.append(he[0])
                wts += [W[eid] * bw for _ in range(hsz-1)]

    num_nodes = 0
    for x in hinc:
        for y in x:
            if y > num_nodes:
                num_nodes = y
    num_nodes += 1
    
    A = csr_array((wts, (src, dst)), shape=(num_nodes, num_nodes))
    # A = csr_array((wts, (src, dst)), shape=(num_nodes + len(hinc), num_nodes + len(hinc)))

    return A