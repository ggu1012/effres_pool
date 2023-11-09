from functions import Filter, HyperEdgeScore, get_num_nodes
from utils.hypergraph_conversion import StarW, FullCliqueW, ExpanderCliqueW
from utils.pklz2hinc import pklz_to_incmat
import numpy as np
from numpy.linalg import qr
import argparse
import sys
import os


def HyperEF(hinc:list, L :int = 3, R :float = 0.5):
    """
    L: # of loops for coarsening
    R: Effective resistance threshold for coarsening 
    """
    # get num nodes
    num_nodes = get_num_nodes(hinc)
    num_edges = len(hinc)

    Neff = np.zeros(num_nodes, dtype=float)
    idx_mat = []

    for loop in range(L):
        print(f"Inside loop {loop + 1}")
        # init weight
        W = np.ones(num_edges)
        A = StarW(hinc, W)

        gram_schmidt_iter = 300
        adj_mat = A.tocoo()
        interval = 20
        initial = 0
        num_rand_vecs = 1
        num_filtered_vecs = int((gram_schmidt_iter - initial) / interval)
        num_vecs = num_rand_vecs * num_filtered_vecs

        Eratio = np.zeros((num_edges, num_vecs), dtype=float)
        SV = np.zeros((num_nodes, num_vecs), dtype=float)

        for ii in range(num_rand_vecs):
            rand_vec = (np.random.rand(A.shape[0]) - 0.5) * 2
            # rand_vec = np.load('HyperEF/src/tmp/ariane_rv.npz').flatten()
            sm = Filter(rand_vec, gram_schmidt_iter, adj_mat, num_nodes, interval, num_vecs)
            SV[:, ii * num_filtered_vecs: (ii+1) * num_filtered_vecs] = sm

        SV, _ = qr(SV, mode='reduced') 
        # QR decomposition to make every approx. eigvecs. orthogonal to each other

        # for jj in range(SV.shape[1]):
        #     hscore = HyperEdgeScore(hinc, SV[:,jj])
        #     Eratio[:,jj] = hscore / sum(hscore)

        # SV.shape = (num_nodes, num_vecs)
        hscore = HyperEdgeScore(hinc, SV)
        Eratio = np.divide(hscore, np.sum(hscore, axis=1)[:, np.newaxis])

        E2 = np.sort(Eratio, axis=1)[:,::-1]
        Evec = E2[:,0]

        P = Evec / max(Evec)
        RT = R * max(Evec)
        Nsample = len(np.where(Evec <= RT)[0])
        PosP = np.argsort(P)

        W[PosP[:Nsample]] = W[PosP[:Nsample]] * \
            (1 + np.divide(1, P[PosP[:Nsample]], where=P[PosP[:Nsample]]!=0))
        
        flag = np.zeros(num_nodes, dtype=bool)
        idx = np.zeros(num_nodes, dtype =int)
        Neff_new = []

        # hedge index with higer hedge weight
        Pos = np.argsort(W,)[::-1]

        val = 1
        for ii in range(Nsample):
            nd = np.array(hinc[Pos[ii]])
            fg = flag[nd]
            fd1 = np.nonzero(fg == False)[0]
            if len(fd1) > 1:
                nd = nd[fd1]
                idx[nd] = val
                flag[nd] = True
                val +=1

                new_val = Evec[Pos[ii]] + sum(Neff[nd])
                Neff_new.append(new_val)

        fdz = np.nonzero(idx == 0)[0]
        idx[fdz] = np.arange(val-1, val + len(fdz) -1)

        Neff_new.append(Neff[fdz])
        idx_mat.append(idx)

        hinc_new = []

        for ii in range(len(hinc)):
            nd = hinc[ii]
            nd_new = set(idx[nd])
            hinc_new.append(tuple(sorted(nd_new)))

        hinc_new = np.array(hinc_new, dtype=object)
        _, fdnu = np.unique(hinc_new, return_index=True)

        singleton = []
        for i, ii in enumerate(fdnu):
            if len(hinc_new[ii]) == 1:
                singleton.append(i)


        fdnu = np.delete(fdnu, singleton)
        W2 = W[fdnu]
        hinc_new = hinc_new[fdnu]

        ### removing hyperedges with cardinality of 1
        singleton_idx = [i for i,x in enumerate(hinc_new) if len(x) == 1]
        
        hinc_new = np.delete(hinc_new, singleton_idx).tolist()
        W2 = np.delete(W2, singleton_idx).tolist()
    
    return idx_mat, hinc_new


def run_HyperEF():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, help='Coarsening level')
    parser.add_argument('--pklz', type=str, help='Pickle file converted from DEF file')
    args = parser.parse_args()
    
    _, hinc = pklz_to_incmat(args.pklz)
    idx_mat, hinc_new = HyperEF(hinc, args.level)


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    run_HyperEF()