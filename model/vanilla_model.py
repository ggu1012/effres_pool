import dgl.sparse as dglsp
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from torch.nn import BatchNorm1d

"""
Utilize 3-cycle uniform expander graph converted from hypergraph

"""


# dgl udf
def scatter_cat_up(edges):
    return {"cat_feat": torch.hstack([edges.src["x"], edges.dst["x"]])}


class VaGNN(nn.Module):
    def __init__(self, gnn_dims, mlp_dims, device="cpu"):
        super().__init__()

        assert gnn_dims[-1] == mlp_dims[0]

        self.act = nn.LeakyReLU()
        self.final_act = nn.Tanh()
        floor_step = [
            # dglnn.GATConv(gnn_dims[i], gnn_dims[i + 1], num_heads=1, feat_drop=0.5).to(
            #     device
            # )
            dglnn.SAGEConv(gnn_dims[i], gnn_dims[i + 1], "mean", feat_drop=0.5).to(device)
            for i in range(len(gnn_dims) - 1)
        ] 

        self.floor_step = nn.ModuleList(floor_step)

        print(self.floor_step)
        mlp = [
            nn.Linear(mlp_dims[i], mlp_dims[i+1]).to(device)
            for i in range(len(mlp_dims) - 1)
        ]
        self.mlp = nn.ModuleList(mlp)

        print(self.mlp)
        
        self.dropout = nn.Dropout(0.5).to(device)

    def forward(self, gr, X):
        sgr = dgl.edge_type_subgraph(gr, [('lv0', 'to', 'lv0')])
        
        xx = X
        for i in range(len(self.floor_step)):
            xx = self.floor_step[i](sgr, xx)
            xx = self.act(xx)
            xx = self.dropout(xx)

        sgr.ndata['x1'], sgr.ndata['x2']  = torch.hsplit(xx, 2)
        # message passing to net node
        sg = dgl.edge_type_subgraph(gr, ['conn'])
        sg.update_all(fn.copy_u('x1', 'm'), fn.max('m', 'y_max'))
        sg.update_all(fn.copy_u('x2', 'm'), fn.max('m', 'y_min'))
        xx = torch.cat([sg.ndata['y_max'].pop('net'), sg.ndata['y_min'].pop('net')], dim=1)

        for i in range(len(self.mlp)-1):
            xx = self.mlp[i](xx)
            xx = self.act(xx)
            xx = self.dropout(xx)
        xx = self.mlp[-1](xx)
        xx = self.final_act(xx)
        return xx
