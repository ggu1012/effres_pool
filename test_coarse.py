import dgl.sparse as dglsp
import gzip
import pickle
import numpy as np
from scipy.sparse import csr_array
from utils.hypergraph_conversion import pklz_to_incmat
import torch
from utils.HyperEF.HyperEF import HyperEF


top = 'aes_cipher_top'
with gzip.open(f'../DREAMPlace/install/dataset/{top}/{top}_0.7_1.0.icc2.pklz') as f:
    xx = pickle.load(f)


H, net2node, mapper = pklz_to_incmat(xx)
idx_mat, xx = HyperEF(net2node, 3)

