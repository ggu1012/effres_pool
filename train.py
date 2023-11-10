import torch
import dgl
import gzip
import pickle

from model.hmodel import HGNN
from utils.HyperEF.HyperEF import HyperEF
from utils.hypergraph_conversion import *
from utils.functions import *

level = 3

top = 'ariane'
with gzip.open(f'../DREAMPlace/install/dataset/{top}/{top}_0.7_1.0.icc2.pklz') as f:
    dataset = pickle.load(f)

H, net2node, _, _ = pklz_to_incmat(dataset)
idx_mat, new_net2nodes = HyperEF(net2node, level)

net2nodes = [net2node, *new_net2nodes]
Hs = [lil_to_dglsp(n2n) for n2n in net2nodes]

rows = idx_mat
cols = [np.arange(len(x)) for x in idx_mat]

# ASM.shape = (num_clusters, num_nodes)-
ASMs = [coo_to_dglsp(rows[i], cols[i]) for i in range(len(idx_mat))]

print([x.shape for x in ASMs])
print([x.shape for x in Hs])

###### !!!! CONVERT TO DGL SPARSE FOR ALL SPARSE MATRICES 
###### ALL MATRICES ARE ASSUMED TO BE IN DGLSP FORMAT AFTER THIS LINE

in_dim = 8
hidden_dims = [32, 128, 256, 256, 128, 32]
out_dim = 8

print("Initialize model")
model = HGNN(Hs, ASMs, in_dim, hidden_dims, out_dim, 'cuda:1')
X = torch.rand(Hs[0].shape[0], in_dim).to('cuda:1')


model.train()
print("Run model")
Y = model(X)
yy = 5 - Y.sum()
yy.backward()