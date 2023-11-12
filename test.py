import torch
import gzip
import pickle
import os
from torch import nn
from glob import glob
import re
from dgl.dataloading import GraphDataLoader
# from model.hmodel import HGNN
from model.exmodel import EXGNN
from utils.HyperEF.HyperEF import HyperEF
from utils.hypergraph_conversion import *
from utils.functions import *

#### CONFIG ####
main_level = 2
sub_level = 2
batch_size = 4
top = 'RocketTile'
device = 'cuda:0'
################


print("Initialize model")
model = torch.load('model_dict_s2')
loss_fn = nn.MSELoss()
dataset_files = glob(f'dataset/y_HPWL/*_raw.pt')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

test_set = []
for i, ds in enumerate(dataset_files):
    if top in ds:
        test_set.append(ds)

xs = []
ys = []
graphs = []
for dataset in test_set:
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

        gr = multi_level_expander_graph(lil_to_dglsp(net2node), net2nodes, ASMs, 3, 'cpu')        
        dgl.save_graphs(graph_fn, [gr])
    else:
        gr = dgl.load_graphs(graph_fn)[0][0].to('cpu')
    x = torch.load(f'dataset/x_node_feature/{top}_{cu}_{ca}.pt').to(torch.float)

    ## HPWL MOD
    y = torch.load(f'dataset/y_HPWL/{basen}').to(torch.float)
    y = y / y.sum()
    y = torch.log10(y)
    ##

    xs.append(x)
    ys.append(y)
    graphs.append(gr)
tmp = CustomDataset(xs, ys, graphs)
test_dataloader = GraphDataLoader(tmp, batch_size = batch_size, shuffle= True, collate_fn=xcollate_fn)

# with profile() as prof:
for x, y, gr in test_dataloader:
    print("Run model")
    x, y, gr = x.to(device), y.to(device), gr.to(device)
    model.eval()
    with torch.no_grad():
        Y = model(gr, x)
    Y = Y.flatten()
    loss = loss_fn(Y, y)
    print(f"loss: {loss.item()}")
    torch.save(y.detach().cpu(), f'results/{top}.m{main_level}.s{sub_level}.gt.pt')
    torch.save(Y.detach().cpu(), f'results/{top}.m{main_level}.s{sub_level}.result.pt')

# print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))