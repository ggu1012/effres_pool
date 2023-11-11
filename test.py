import torch
import gzip
import pickle
import os
from torch import nn
from glob import glob
import re
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


print("Initialize model")
model = torch.load('model_dict')
loss_fn = nn.MSELoss()
dataset_files = glob(f'dataset/x_node_feature/*.pt')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

top = 'jpeg_encoder'
test_set = []
for i, ds in enumerate(dataset_files):
    if top in ds:
        test_set.append(ds)

for ep, dataset in enumerate(test_set):
    basen = os.path.basename(dataset)
    rem = re.match(r'(.*)_(.*)_(.*)\.pt$', basen)
    top, cu, ca = rem.group(1), rem.group(2), rem.group(3)
    
    with gzip.open(f'../DREAMPlace/install/dataset/{top}/{top}_{cu}_{ca}.icc2.pklz') as f:
        dataset = pickle.load(f)
    H, net2node, net_map, node_map = pklz_to_incmat(dataset)
    print(f"Top: {top}, #Nodes: {H.shape[0]}, #Edges: {H.shape[1]}")
    chunkname = f'graph/{top}_{cu}_{ca}.m{main_level}.s{sub_level}.pkl'

    with gzip.open(chunkname, 'rb') as f:
        Hs, idx_mat, net2nodes, net_map, node_map = pickle.load(f)

    rows = idx_mat
    cols = [np.arange(len(x)) for x in idx_mat]
    ASMs = [coo_to_dglsp(rows[i], cols[i]) for i in range(len(idx_mat))] # ASM.shape = (num_clusters, num_nodes)

    gr, H = (multi_level_expander_graph(net2nodes, ASMs, 3, device), \
                   lil_to_dglsp(net2node).to(device))
    
    x = torch.load(f'dataset/x_node_feature/{basen}').to(torch.float).to(device)
    y = torch.load(f'dataset/y_HPWL/{top}_{cu}_{ca}_raw.pt').to(torch.float).to(device)
    y = torch.log10(y)


    # with profile() as prof:
    print("Run model")
    model.eval()
    with torch.no_grad():
        Y = model(gr, H, x)
    Y = Y.flatten()
    loss = loss_fn(Y, y)
    print(f"loss: {loss.item()}")

    torch.save(Y.detach().cpu(), f'results/{top}_{cu}_{ca}.m{main_level}.s{sub_level}.result.pt')

# print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))