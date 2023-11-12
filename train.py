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
from model.vanilla_model import VaGNN

from utils.HyperEF.HyperEF import HyperEF
from utils.hypergraph_conversion import *
from utils.functions import *

#### CONFIG ####
main_level = 2
sub_level = 2
tail = 2
num_labels = 10
val_set = ['ariane', 'RocketTile']
test_set = ['bsg_chip']
device = 'cuda:0'
################


# top = 'mempool_tile_wrap'
# cu, ca = 0.7, 1.0
# with gzip.open(f'../DREAMPlace/install/dataset/{top}/{top}_{cu}_{ca}.icc2.pklz') as f:
#     dataset = pickle.load(f)
# H, net2node, net_map, node_map = pklz_to_incmat(dataset)
# print(f"Top: {top}, #Nodes: {H.shape[0]}, #Edges: {H.shape[1]}")
# chunkname = f'graph/pre_process/{top}_{cu}_{ca}.m{main_level}.s{sub_level}.pkl'

# with gzip.open(chunkname, 'rb') as f:
#     Hs, idx_mat, net2nodes, net_map, node_map = pickle.load(f)

# rows = idx_mat
# cols = [np.arange(len(x)) for x in idx_mat]
# ASMs = [coo_to_dglsp(rows[i], cols[i]) for i in range(len(idx_mat))] # ASM.shape = (num_clusters, num_nodes)

# xxxx = ExpanderCliqueW(net2node, 3, 54321)
# print(dgl.to_bidirected(dgl.graph(('csr', (xxxx.indptr, xxxx.indices, [])))))
# # gr = multi_level_expander_graph(lil_to_dglsp(net2node), net2nodes, ASMs, 3, 'cpu') 

# exit()



gnn_dims = [5, 64, 256, 256, 64, 32]
mlp_dims = [32, 64, 64, 1]
assert len(gnn_dims) == 2 * main_level + 2


# dataset_files = glob(f'dataset/x_node_feature/*.pt')
dataset_files = glob(f'dataset/y_HPWL/*_raw.pt')

train_dataset = []
val_dataset = []
test_dataset = []

for i, ds in enumerate(dataset_files):
    if re.match(f'.*({"|".join(val_set)}).*', ds):
        val_dataset.append(ds)
    elif re.match(f'.*({"|".join(test_set)}).*', ds):
        test_dataset.append(ds)
    else:
        train_dataset.append(ds)

dataloaders = []
batch_sizes = [4, 4, 2]
for i, mode in enumerate([train_dataset, val_dataset, test_dataset]):
    xs = []
    ys = []
    graphs = []
    for dataset in mode:
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
        # x = torch.rand((gr.num_nodes('lv0'), 5), dtype=torch.float)

        ## HPWL MOD
        y = torch.load(f'dataset/y_HPWL/{basen}').to(torch.float)
        y = y / y.sum()
        y = torch.log10(y)
        ##

        xs.append(x)
        ys.append(y)
        graphs.append(gr)
    tmp = CustomDataset(xs, ys, graphs)
    dataloaders.append(GraphDataLoader(tmp, batch_size = batch_sizes[i], shuffle= True, collate_fn=xcollate_fn))

print("Initialize model")
# model = EXGNN(gnn_dims, mlp_dims, main_level, device)
model = VaGNN(gnn_dims, mlp_dims, device)
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5)

# dataloaders = [train, val, test]
for epoch in range(100):
    total_loss = 0

    ## Train
    model.train()
    losses = []
    for x, y, gr in dataloaders[0]:
        x,y,gr = x.to(device),y.to(device),gr.to(device)
        Y = model(gr, x)
        Y = Y.flatten()
        loss = F.smooth_l1_loss(Y, y)

        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

    print(f"Epoch {epoch + 1} Train - Total avg. loss: {sum(losses) / len(losses)}")
    writer.add_scalar("Loss/train", sum(losses) / len(losses), epoch)

    ## Validation
    if (epoch + 1) % 10 == 0:
        model.eval()
        losses = []
        for i, (x, y, gr) in enumerate(dataloaders[1]):
            gr, x, y = gr.to(device), x.to(device), y.to(device)
            with torch.no_grad():
                Y = model(gr, x)
            Y = Y.flatten()
            loss = F.smooth_l1_loss(Y, y)
            losses.append(loss)
            torch.save(y.detach().cpu(), f'results/val_m{main_level}.s{sub_level}_e{epoch+1}_b{i}.gt.pt')
            torch.save(Y.detach().cpu(), f'results/val_m{main_level}.s{sub_level}_e{epoch+1}_b{i}.result.pt')
    
        print(f" ==== Epoch {epoch + 1} Val - Total avg. loss: {sum(losses) / len(losses)}")

torch.save(model, './model_dict_s2')

## Test
model.eval()
losses = []
for i, (x, y, gr) in enumerate(dataloaders[2]):
    gr, x, y = gr.to(device), x.to(device), y.to(device)
    with torch.no_grad():
        Y = model(gr, x)
    Y = Y.flatten()
    loss = F.mse_loss(Y, y)
    losses.append(loss)
    torch.save(y.detach().cpu(), f'results/test_{top}.m{main_level}.s{sub_level}_b{i}.gt.pt')
    torch.save(Y.detach().cpu(), f'results/test_{top}.m{main_level}.s{sub_level}_b{i}.result.pt')

print(f"Test - Total avg. loss: {sum(losses) / len(losses)}")




# print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))