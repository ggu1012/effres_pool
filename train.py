import torch
import gzip
import pickle
import os
from torch import nn
# from torch.profiler import profile, record_function, ProfilerActivity
from torch.autograd.profiler import profile

# from model.hmodel import HGNN
from model.exmodel import EXGNN

from utils.HyperEF.HyperEF import HyperEF
from utils.hypergraph_conversion import *
from utils.functions import *

main_level = 2
sub_level = 2
device = 'cuda:0'

top = 'bsg_chip'
with gzip.open(f'../DREAMPlace/install/dataset/{top}/{top}_0.7_1.0.icc2.pklz') as f:
    dataset = pickle.load(f)

H, net2node, net_map, node_map = pklz_to_incmat(dataset)

print(f"Top: {top}, #Nodes: {H.shape[0]}, #Edges: {H.shape[1]}")

chunkname = f'graph/{top}.m{main_level}.s{sub_level}.npz'
if not os.path.exists(chunkname):
    idx_mats = []
    net2nodes_collection = []
    idx_mat, new_net2nodes = HyperEF(net2node, main_level, sub_level, 42312)
    net2nodes = [net2node, *new_net2nodes]

    with gzip.open(chunkname, 'wb') as f:
        pickle.dump((net2nodes, idx_mat), f)
else:
    with gzip.open(chunkname, 'rb') as f:
        net2nodes, idx_mat = pickle.load(f)

rows = idx_mat
cols = [np.arange(len(x)) for x in idx_mat]

# ASM.shape = (num_clusters, num_nodes)
ASMs = [coo_to_dglsp(rows[i], cols[i]) for i in range(len(idx_mat))]

###### !!!! CONVERT TO DGL SPARSE FOR ALL SPARSE MATRICES 
###### ALL MATRICES ARE ASSUMED TO BE IN DGLSP FORMAT AFTER THIS LINE

dims = [5, 32, 128, 128, 32, 8]
assert len(dims) == 2 * main_level + 2

print("Initialize model")
model = EXGNN(net2nodes, ASMs, dims, device)
x = torch.load(f'dataset/x_node_feature/{top}_0.7_1.0.pt')[node_map].to(device).to(torch.float)
y = torch.load(f'dataset/y_HPWL/{top}_0.7_1.0.pt').to(device).to(torch.float)

# with profile() as prof:
print("Run model")
loss = nn.MSELoss()
H = lil_to_dglsp(net2node)
print(H.shape)
H = H.to(device)
Y = model(x)

loss_ = loss(y, (H.T @ Y).flatten())
loss_.backward()

# print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))