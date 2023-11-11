import torch
import gzip
import pickle
import os
from torch import nn
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from glob import glob
import re
# from torch.profiler import profile, record_function, ProfilerActivity
from torch.autograd.profiler import profile
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# from model.hmodel import HGNN
from model.exmodel import EXGNN

from utils.HyperEF.HyperEF import HyperEF
from utils.hypergraph_conversion import *
from utils.functions import *

#### CONFIG ####
main_level = 2
sub_level = 2
tail = 2
num_labels = 10
device = 'cuda:0'
################



gnn_dims = [5, 64, 256, 256, 64, 32]
mlp_dims = [32, 64, 64, 1]
assert len(gnn_dims) == 2 * main_level + 2

print("Initialize model")
model = EXGNN(gnn_dims, mlp_dims, main_level, device)
model.train()

# dataset_files = glob(f'dataset/x_node_feature/*.pt')
dataset_files = glob(f'dataset/y_HPWL/*_raw.pt')
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)

rem = []
for i, ds in enumerate(dataset_files):
    if 'bsg_chip' in ds:
        rem.append(i)
# dataset_files = np.delete(np.array(dataset_files, dtype=object), rem)
# rem = []
# for i, ds in enumerate(dataset_files):
#     if 'RocketTile' in ds:
#         rem.append(ds)
# dataset_files = rem

graphs = []
xs = []
ys = []
for dataset in dataset_files:
    basen = os.path.basename(dataset)
    rem = re.match(r'(.*)_(.*)_(.*)_.*\.pt$', basen)
    top, cu, ca = rem.group(1), rem.group(2), rem.group(3)
    graph_fn = f'graph/dgl/{top}_{cu}_{ca}.bin'
    if not os.path.exists(graph_fn):
        with gzip.open(f'../DREAMPlace/install/dataset/{top}/{top}_{cu}_{ca}.icc2.pklz') as f:
            dataset = pickle.load(f)
        H, net2node, net_map, node_map = pklz_to_incmat(dataset)
        print(f"Top: {top}, #Nodes: {H.shape[0]}, #Edges: {H.shape[1]}")
        chunkname = f'graph/pre_process/{top}_{cu}_{ca}.m{main_level}.s{sub_level}.pkl'

        with gzip.open(chunkname, 'rb') as f:
            Hs, idx_mat, net2nodes, net_map, node_map = pickle.load(f)

        rows = idx_mat
        cols = [np.arange(len(x)) for x in idx_mat]
        ASMs = [coo_to_dglsp(rows[i], cols[i]) for i in range(len(idx_mat))] # ASM.shape = (num_clusters, num_nodes)

        gr = multi_level_expander_graph(lil_to_dglsp(net2node), net2nodes, ASMs, 3, device)        
        dgl.save_graphs(graph_fn, [gr])
    else:
        gr = dgl.load_graphs(graph_fn)[0][0].to(device)
    x = torch.load(f'dataset/x_node_feature/{top}_{cu}_{ca}.pt').to(torch.float).to(device)

    ## HPWL MOD
    y = torch.load(f'dataset/y_HPWL/{basen}').to(torch.float).to(device)
    y = y / y.sum()
    y = torch.log10(y)
    ##

    xs.append(x)
    ys.append(y)
    graphs.append(gr)

my_dataset = CustomDataset(xs, ys, graphs)
dataloader = GraphDataLoader(my_dataset, batch_size = 1, shuffle= True, collate_fn=xcollate_fn)

for epoch in range(100):
    total_loss = 0
    for x, y, gr in dataloader:
        Y = model(gr, x)
        Y = Y.flatten()
        loss = F.mse_loss(Y, y)

        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    print(f"Total loss: {total_loss}")
    writer.add_scalar("Loss/train", total_loss, epoch)


torch.save(model, './model_dict_s2')


# print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))