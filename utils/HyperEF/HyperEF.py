from utils.HyperEF.functions import *
from utils.hypergraph_conversion import StarW, FullCliqueW, ExpanderCliqueW, pklz_to_incmat
import numpy as np
from numpy.linalg import qr
import argparse
import sys
import os


def HyperEF(hinc: list, L :int = 3, R :float = 0.5):
    """
    L: # of loops for coarsening
    R: Effective resistance threshold for coarsening 
    """
    # get num nodes
    num_nodes = get_num_nodes(hinc)
    Neff = np.zeros(num_nodes, dtype=float)

    idx_mats = []
    hinc_news = []

    for loop in range(L):
        print(f"Inside loop {loop + 1}")

        # get num nodes
        num_nodes = get_num_nodes(hinc)
        num_edges = len(hinc)

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
            sm = Filter(rand_vec, gram_schmidt_iter, adj_mat, num_nodes, interval, num_vecs)
            SV[:, ii * num_filtered_vecs: (ii+1) * num_filtered_vecs] = sm

        SV, _ = qr(SV, mode='reduced') 
        # QR decomposition to make every approx. eigvecs. orthogonal to each other

        # for jj in range(SV.shape[1]):
        #     hscore = orig_HyperEdgeScore(hinc, SV[:,jj])
        #     Eratio[:,jj] = hscore / sum(hscore)

        # SV.shape = (num_nodes, num_vecs)
        hscore = HyperEdgeScore(hinc, SV)
        Eratio = np.divide(hscore, np.sum(hscore, axis=0))

        print(Eratio.shape)
        if loop == 0:
            np.save("../HyperEF_julia/src/tmp/aes_cipher_top_py", Eratio)

        E2 = np.sort(Eratio, axis=1)[:,::-1]
        Evec = E2[:,0]

        P = Evec / max(Evec)
        RT = R * max(Evec)
        Nsample = len(np.where(Evec <= RT)[0])
        PosP = np.argsort(P)

        W[PosP[:Nsample]] = W[PosP[:Nsample]] * \
            (1 + np.divide(1, P[PosP[:Nsample]], where=P[PosP[:Nsample]]!=0))
        
        flag = np.zeros(num_nodes, dtype=bool)
        idx = -1 * np.ones(num_nodes, dtype =int)
        Neff_new = []

        # hedge index with higher hedge weight
        Pos = np.argsort(W,)[::-1]

        val = 0
        for ii in range(Nsample):
            nd = np.array(hinc[Pos[ii]], dtype=int)
            fg = flag[nd]
            fd1 = np.nonzero(fg == False)[0]
            if len(fd1) > 1:
                nd = nd[fd1]
                idx[nd] = val
                flag[nd] = True
                val +=1

                new_val = Evec[Pos[ii]] + sum(Neff[nd])
                Neff_new.append(new_val)

        fdz = np.nonzero(idx == -1)[0]
        idx[fdz] = np.arange(val, val + len(fdz))

        Neff_new = np.append(Neff_new, Neff[fdz])
        print(len(Neff_new), len(fdz), len(Neff))
        idx_mats.append(idx)

        hinc_new = []

        for ii in range(len(hinc)):
            nd = np.array(hinc[ii], dtype=int)
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
        W_new = np.delete(W2, singleton_idx).tolist()
        hinc_news.append(hinc_new)

        hinc = hinc_new
        Neff = Neff_new
        W = W_new

    
    return idx_mats, hinc_news


def run_HyperEF(pklz, level):
    
    _, hinc = pklz_to_incmat(pklz)
    idx_mat, hinc_new = HyperEF(hinc, level)

    return idx_mat, hinc_new
