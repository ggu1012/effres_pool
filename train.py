import torch
import gzip
import pickle
import os
from torch import nn
import torch.nn.functional as F
import ray
from dgl.dataloading import GraphDataLoader
from glob import glob
import re

# from torch.profiler import profile, record_function, ProfilerActivity
from torch.autograd.profiler import profile
from torch.utils.tensorboard import SummaryWriter


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
num_labels = 20
labeled = False
val_set = ["bsg_chip", "RocketTile"]
test_set = ["ariane"]
device = "cuda:0"
################


gnn_dims = [8, 64, 256, 256, 64, 32]
x_net_dim = 3 ## x edge feature; net_degree, core_utilization, core_aspect
tail_dims = -1
mlp_dims = [32 + x_net_dim, 64]
mlp_dims += [num_labels] if labeled == True else [1]

assert len(gnn_dims) == 2 * main_level + 2

@ray.remote
def get_dataset_xy(dataset):
    basen = os.path.basename(dataset)
    rem = re.match(r"(.*)_(.*)_(.*)_.*\.pt$", basen)
    top, cu, ca = rem.group(1), rem.group(2), rem.group(3)
    graph_fn = f"graph/dgl/{top}_{cu}_{ca}.bin"
    chunkname = f"graph/pre_process/{top}_{cu}_{ca}.m{main_level}.s{sub_level}.pkl"
    with gzip.open(chunkname, "rb") as f:
        Hs, idx_mat, net2nodes, net_map, node_map = pickle.load(f)

    original_H = lil_to_dglsp(net2nodes[0])

    if not os.path.exists(graph_fn):
        with gzip.open(
            f"../DREAMPlace/install/dataset/{top}/{top}_{cu}_{ca}.icc2.pklz"
        ) as f:
            dataset = pickle.load(f)

        rows = idx_mat
        cols = [np.arange(len(x)) for x in idx_mat]
        ASMs = [
            coo_to_dglsp(rows[i], cols[i]) for i in range(len(idx_mat))
        ]  # ASM.shape = (num_clusters, num_nodes)

        gr = multi_level_expander_graph(original_H, net2nodes, ASMs, 3, "cpu")
        dgl.save_graphs(graph_fn, [gr])
    else:
        gr = dgl.load_graphs(graph_fn)[0][0].to("cpu")

    x = torch.load(f"dataset/x_node_feature/{top}_{cu}_{ca}.pt").to(
        torch.float
    )  # already cut-off
    # x = torch.rand((gr.num_nodes('lv0'), 5), dtype=torch.float)

    ## x node feature;
    gr.ndata["x"] = {"lv0": x}

    ## x edge feature; net_degree, core_utilization, core_aspect
    net_feat = torch.hstack([
        original_H.sum(dim=0).reshape(-1, 1),
        torch.full((original_H.shape[1], 1), float(cu)),
        torch.full((original_H.shape[1], 1), float(ca)),
    ])
    gr.ndata['x_net'] = {'net': net_feat}

    ## HPWL MOD
    y = torch.load(f"dataset/y_HPWL/{basen}").to(torch.float)[net_map]
    y = torch.log10(y)

    if labeled == True:
        bins = torch.histogram(y, bins=num_labels)[1]
        y = torch.bucketize(y, bins, right=True) - 1
        y[y == num_labels] = num_labels - 1
    else:
        # y = y - y.min() # 0 ~
        # y = y - y.max() # ~ 0
        _max = y.max()
        _min = y.min()
        # y = (((y - _min) / (_max - _min)) -0.5) * 2 # -1 ~ 1
        y = (y - _min) / (_max - _min)  # 0 ~ 1
    ##

    return gr, y


if labeled == True:
    print("Running with LABELED HPWL")

dataset_files = glob(f"dataset/y_HPWL/*_raw.pt")
train_dataset = []
val_dataset = []
test_dataset = []

ray.init(num_cpus=32)

for i, ds in enumerate(dataset_files):
    if re.match(f'.*({"|".join(val_set)}).*', ds):
        val_dataset.append(ds)
    elif re.match(f'.*({"|".join(test_set)}).*', ds):
        test_dataset.append(ds)
    else:
        train_dataset.append(ds)

print("Initialize model")

# loss_fn = nn.MSELoss()
# loss_fn = nn.SmoothL1Loss()
loss_fn = lambda input, target, wts: (wts * (input - target) ** 2).mean()

model = EXGNN(gnn_dims, mlp_dims, x_net_dim, tail_dims, main_level, device)
# model = VaGNN(gnn_dims, mlp_dims, device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=5e-2)

dataloaders = []
batch_sizes = [4, 1, 1]
for i, m_dataset in enumerate([train_dataset, val_dataset, test_dataset]):
    xs = []
    ys = []
    graphs = []
    jobs = [get_dataset_xy.remote(x) for x in m_dataset]
    chunk = ray.get(jobs)
    for gr, y in chunk:
        ys.append(y)
        graphs.append(gr)
    tmp = CustomDataset(graphs, ys)
    dataloaders.append(
        GraphDataLoader(
            tmp, batch_size=batch_sizes[i], shuffle=True, collate_fn=xcollate_fn
        )
    )

writer = SummaryWriter()

# dataloaders = [train, val, test]
for epoch in range(300):
    total_loss = 0

    ## Train
    model.train()
    losses = []
    for gr, y in dataloaders[0]:
        y, gr = y.to(device), gr.to(device)
        Y = model(gr)
        if labeled == True:
            y_counts = torch.unique(y, return_counts=True)[1]
            # y_wts = 1 - y_counts / y_counts.sum()
            loss = F.cross_entropy(Y, y, label_smoothing=0.2)
        else:
            Y = Y.flatten()
            wts = y
            loss = loss_fn(Y, y, wts)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

    print(f"Epoch {epoch + 1} Train - Total avg. loss: {sum(losses) / len(losses)}")
    writer.add_scalar("Loss/train", sum(losses) / len(losses), epoch + 1)

    ## Validation
    if (epoch + 1) % 10 == 0:
        torch.save(
            y.detach().cpu(),
            f"results/train_m{main_level}.s{sub_level}_e{epoch+1}.gt.pt",
        )
        torch.save(
            Y.detach().cpu(),
            f"results/train_m{main_level}.s{sub_level}_e{epoch+1}.result.pt",
        )

        model.eval()
        losses = []
        for i, (gr, y) in enumerate(dataloaders[1]):
            gr, y = gr.to(device), y.to(device)
            with torch.no_grad():
                Y = model(gr)
            if labeled == True:
                y_counts = torch.unique(y, return_counts=True)[1]
                # y_wts = 1 - y_counts / y_counts.sum()
                loss = F.cross_entropy(Y, y, label_smoothing=0.2)
            else:
                Y = Y.flatten()
                wts = y
                loss = loss_fn(Y, y, wts)
            losses.append(loss)
            torch.save(
                y.detach().cpu(),
                f"results/val_m{main_level}.s{sub_level}_e{epoch+1}_b{i}.gt.pt",
            )
            torch.save(
                Y.detach().cpu(),
                f"results/val_m{main_level}.s{sub_level}_e{epoch+1}_b{i}.result.pt",
            )
        writer.add_scalar("Loss/val", sum(losses) / len(losses), epoch + 1)
        print(
            f" ==== Epoch {epoch + 1} Val - Total avg. loss: {sum(losses) / len(losses)}"
        )
        torch.save(model, f"saved_models/model_dict_e{epoch+1}")

## Test
model.eval()
losses = []
for i, (y, gr) in enumerate(dataloaders[2]):
    gr, y = gr.to(device), y.to(device)
    with torch.no_grad():
        Y = model(gr)
    if labeled == True:
        y_counts = torch.unique(y, return_counts=True)[1]
        # y_wts = 1 - y_counts / y_counts.sum()
        loss = F.cross_entropy(Y, y, label_smoothing=0.2)
    else:
        Y = Y.flatten()
        wts = y
        loss = loss_fn(Y, y, wts)
    losses.append(loss)
    torch.save(y.detach().cpu(), f"results/test.m{main_level}.s{sub_level}_b{i}.gt.pt")
    torch.save(
        Y.detach().cpu(), f"results/test.m{main_level}.s{sub_level}_b{i}.result.pt"
    )

print(f"Test - Total avg. loss: {sum(losses) / len(losses)}")
