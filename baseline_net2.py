import kahypar
from utils.functions import *
import torch
import dgl
import dgl.nn as dglnn
import kahypar
from glob import glob
from scipy.sparse import isspmatrix_csr, dok_array
import ray

# from scipy.sparse import


def net2_graph(dname, pkl):
    ## Variation of utils.hypergraph_conversion.driver2load
    H, orig_net2node, node_mapper, net_mapper = pklz_to_incmat(
        dname, pkl
    )  # @@ BRAINWASH @@

    pin2node = np.array(pkl["pin_info"]["pin2net_map"])  # @@ BRAINWASH @@
    pin2net = np.array(pkl["pin_info"]["pin2node_map"])  # @@ BRAINWASH @@
    pin_direction = np.array(pkl["pin_info"]["pin_direct"])
    H = H.T

    rev_net_mapper = {}
    for i, nd in enumerate(net_mapper):
        rev_net_mapper[nd] = i
    rev_node_mapper = {}
    for i, nd in enumerate(node_mapper):
        rev_node_mapper[nd] = i

    num_nodes, num_nets = H.shape
    net2node = [[] for _ in range(num_nets)]

    # find and remove net with no INPUT pin
    net_in_flag = np.zeros(num_nets, dtype=bool)
    net_out_flag = np.zeros(num_nets, dtype=bool)
    net_in = [[] for _ in range(num_nets)]
    net_out = [[] for _ in range(num_nets)]
    for pin in range(len(pin2node)):
        v = pin2node[pin]
        e = pin2net[pin]

        if v not in node_mapper or e not in net_mapper:
            continue
        dir = pin_direction[pin]
        new_e = rev_net_mapper[e]
        new_v = rev_node_mapper[v]
        net2node[new_e].append(new_v)
        if dir == b"INPUT":
            net_in_flag[new_e] = True
            net_in[new_e].append(new_v)
        if dir == b"OUTPUT":
            net_out_flag[new_e] = True
            net_out[new_e].append(new_v)

    net_in = np.array(net_in, dtype=object)
    net_out = np.array(net_out, dtype=object)
    remove_nets = np.nonzero(np.bitwise_and(net_in_flag, net_out_flag) == 0)[0]
    new_net_mapper = np.delete(net_mapper, remove_nets)
    net_in = np.delete(net_in, remove_nets)
    net_out = np.delete(net_out, remove_nets)


    H = H[:, np.delete(np.arange(H.shape[1]), remove_nets)]
    void_node = np.nonzero(H.sum(axis=1) == 0)[0]

    if len(void_node) > 0:
        new_node_mapper = np.delete(node_mapper, void_node)
        H = H[np.delete(np.arange(H.shape[0]), remove_nets)]
    else:
        new_node_mapper = node_mapper

    H = H.tocsc()
    indptr = H.indptr
    indices = H.indices

    ### net2node generation
    # remove void node
    net2node = []
    for i in range(len(indptr) - 1):
        net = indptr[i]
        nodes = indices[indptr[i] : indptr[i + 1]]
        net2node.append(list(nodes))

    # src_dst = {}
    row = []
    col = []
    A = dok_array((num_nodes, num_nodes), dtype=int)
    for i in range(len(net2node)):
        srcs = net_in[i]
        dsts = net_out[i]
        for x in srcs:
            for y in dsts:
                if A[x,y] != 0:
                    continue
                else:
                    A[x, y] = i
    # new_net_mapper == cell mapper
    # new_node_mapper == net mapper
    A = A.tocsr()
    return A.tocsr(), net2node, H, new_net_mapper, new_node_mapper


def cell_edge_partition(H, k):
    H = H.T.tocsc()  # ExV to VxE
    hypergraph = kahypar.Hypergraph(
        H.shape[0], H.shape[1], list(H.indptr), list(H.indices), k
    )
    context = kahypar.Context()
    context.setK(k)
    context.loadINIconfiguration("baselines/net2/km1_kKaHyPar_sea20.ini")
    context.setEpsilon(0.03)
    kahypar.partition(hypergraph, context)

    blockID = []
    for x in hypergraph.nodes():
        blockID.append(hypergraph.blockID(x))

    return np.array(blockID, dtype=int)


def measurediff(A, cidx1, nidx1, cidx2, nidx2, netpart, cellpart):
    f0 = 1 - (cellpart[cidx1] == cellpart[cidx2])

    nidx1_cell_neighs = []
    nidx2_cell_neighs = []
    for _A in A, A.T.tocsr():
        nidx1_cell_neighs.append(_A.data[_A.indptr[nidx1] : _A.indptr[nidx1 + 1]])
        nidx2_cell_neighs.append(_A.data[_A.indptr[nidx2] : _A.indptr[nidx2 + 1]])
    nidx1_cell_neighs = np.concatenate(nidx1_cell_neighs)
    nidx2_cell_neighs = np.concatenate(nidx2_cell_neighs)
    Pnidx1 = np.unique(cellpart[nidx1_cell_neighs])
    Pnidx2 = np.unique(cellpart[nidx2_cell_neighs])
    P1_not_2 = np.setdiff1d(Pnidx1, Pnidx2)
    P2_not_1 = np.setdiff1d(Pnidx2, Pnidx1)
    f1 = len(P1_not_2) / len(Pnidx1) + len(P2_not_1) / len(Pnidx2)
    f2 = 1 - (netpart[nidx1] == netpart[nidx2])

    return f0, f1, f2


def gen_edge_feature(A, netpart, cellpart):
    efeat = {}
    efeat = []
    eid = []
    for nk in range(A.shape[0]):
        for _A in A, A.T:
            nk_neighs = _A.indices[_A.indptr[nk] : _A.indptr[nk + 1]]
            start = 1
            for nb in nk_neighs:
                cbk = _A[nb, nk]
                F0, F1, F2 = [], [], []
                for no in nk_neighs[start:]:
                    cok = _A[no, nk]
                    f0, f1, f2 = measurediff(A, cbk, nb, cok, no, netpart, cellpart)
                    F0.append(f0)
                    F1.append(f1)
                    F2.append(f2)
                f3 = 1 - (netpart[nb] == netpart[nk])
                F0 = np.array(F0, dtype=float)
                F1 = np.array(F1, dtype=float)
                F2 = np.array(F2, dtype=float)
                std_F0 = np.std(F0) if len(F0) > 1 else 0
                std_F1 = np.std(F1) if len(F1) > 1 else 0
                std_F2 = np.std(F2) if len(F2) > 1 else 0

                start += 1
                efeat.append(np.array(
                    [F0.sum(),std_F0, F1.sum(), std_F1, F2.sum(), std_F2, f3]
                ))
                eid.append(np.array([nb, nk]))
    efeat = np.vstack(efeat)
    eids = np.vstack(eid)

    chunk = torch.from_numpy(np.hstack([eids, efeat]))
            
    return chunk

@ray.remote
def one_job(dataset, k_net, k_cell):

    with gzip.open(dataset, 'rb') as f:
        obj = pickle.load(f)
    basen = os.path.basename(dataset)
    top_var = re.match(r'^(.*)\.icc2\.pklz$', basen).group(1)
    graphname = f'baselines/net2/net2_graph/{top_var}.bin'
    mapname = f'baselines/net2/net2_map/{top_var}'
    efeatname = f'baselines/net2/net2_efeat/{top_var}.pt'

    A, net2node, H, net_map, node_map = net2_graph(dataset, obj)
    Net2G = dgl.heterograph(
        {
            ("net", "fanout", "net"): (
                "csr",
                (torch.tensor(A.indptr), torch.tensor(A.indices), []),
            ),
            ("net", "fanin", "net"): (
                "csc",
                (torch.tensor(A.T.indptr), torch.tensor(A.T.indices), []),
            ),
        }, idtype=torch.int64
    )

    dgl.save_graphs(graphname, [Net2G])
    np.save(mapname, np.array([net_map, node_map], dtype=object))

    net_node_partition_assignment = dgl.metis_partition_assignment(
        dgl.to_homogeneous(Net2G), k_net
    ).numpy()
    cell_edge_partition_assignment = cell_edge_partition(H, k_cell)
    efeat_chunk = gen_edge_feature(A, net_node_partition_assignment, cell_edge_partition_assignment)

    torch.save(efeat_chunk, efeatname)
    
    return 1

def main():

    top = ''
    k_net = 500
    k_cell = 100

    if top:
        dsets = glob(f'../DREAMPlace/install/dataset/{top}/*.icc2.pklz')
    else:
        dsets = glob(f'../DREAMPlace/install/dataset/*/*.icc2.pklz')

    ray.init(num_cpus = 32)

    jobs = []
    for dataset in dsets:
        jobs.append(one_job.remote(dataset, k_net, k_cell))
    
    results = ray.get(jobs)
    print(results)



if __name__ == "__main__":
    main()
