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


class EXGNN(nn.Module):
    def __init__(self, gnn_dims, mlp_dims, pool_level, device="cpu"):
        super().__init__()

        assert len(gnn_dims) == 2 * (pool_level + 1)
        assert gnn_dims[-1] == mlp_dims[0]

        self.pool_level = pool_level

        floor_step = [
            dglnn.GATConv(gnn_dims[i], gnn_dims[i + 1], num_heads=1, feat_drop=0.5).to(
                device
            )
            # dglnn.SAGEConv(gnn_dims[i], gnn_dims[i + 1], "mean", feat_drop=0.5).to(device)
            for i in range(self.pool_level + 1)
        ] + [
            dglnn.GATConv(
                gnn_dims[i] + gnn_dims[2 * self.pool_level + 1 - i],
                gnn_dims[i + 1],
                num_heads=1,
                feat_drop=0.5,
            ).to(device)
            # dglnn.SAGEConv(
            #     gnn_dims[i] + gnn_dims[2 * self.pool_level + 1 - i],
            #     gnn_dims[i + 1],
            #     "mean",
            #     feat_drop=0.5,
            # ).to(device)
            for i in range(self.pool_level + 1, 2 * self.pool_level + 1)
        ]

        self.floor_step = nn.ModuleList(floor_step)

        print(self.floor_step)

        mlp = [nn.Linear(gnn_dims[-1], mlp_dims[0]).to(device)]
        mlp += [
            nn.Linear(mlp_dims[i], mlp_dims[i + 1]).to(device)
            for i in range(len(mlp_dims) - 1)
        ]
        self.mlp = nn.ModuleList(mlp)

        print(self.mlp)

        self.dropout = nn.Dropout(0.5).to(device)

    def forward(self, gr, X):
        gr.ndata["x"] = {"lv0": X}

        # Goes down
        #   GNN
        # o ----
        #        \  Pool
        #         \
        #          o

        for i in range(self.pool_level):
            sg = dgl.edge_type_subgraph(gr, [(f"lv{i}", "to", f"lv{i}")])
            xx = gr.ndata["x"].pop(f"lv{i}")
            xx = self.floor_step[i](sg, xx)
            gr.ndata["x"] = {f"lv{i}": xx}

            sg = dgl.edge_type_subgraph(gr, [(f"lv{i}", "downwards", f"lv{i+1}")])
            sg.update_all(fn.copy_u("x", "m"), fn.mean("m", "x"))
            xx = gr.ndata["x"].pop(f"lv{i+1}")
            gr.ndata["x"][f"lv{i+1}"] = F.relu(xx)

        # Bottom
        sg = dgl.edge_type_subgraph(
            gr, [(f"lv{self.pool_level}", "to", f"lv{self.pool_level}")]
        )
        xx = gr.ndata["x"].pop(f"lv{self.pool_level}")
        xx = self.floor_step[self.pool_level](sg, xx)
        gr.ndata["x"][f"lv{self.pool_level}"] = F.relu(xx)

        # Goes up
        #              GNN
        #             ---- o
        #            /
        #   Uplift  /
        #          o

        pl = self.pool_level
        for i in range(pl):
            rev_i = pl - i - 1
            sg = dgl.edge_type_subgraph(gr, [(f"lv{rev_i+1}", "upwards", f"lv{rev_i}")])
            sg.update_all(scatter_cat_up, fn.sum("cat_feat", "x"))
            sg = dgl.edge_type_subgraph(gr, [(f"lv{rev_i}", "to", f"lv{rev_i}")])
            xx = sg.ndata.pop("x")
            xx = self.floor_step[pl + i + 1](sg, xx)
            xx = F.relu(xx)
            sg.ndata["x"] = xx

        # message passing to net node
        sg = dgl.edge_type_subgraph(gr, ["conn"])
        sg.update_all(fn.copy_u("x", "m"), fn.sum("m", "y_pred"))
        xx = sg.ndata["y_pred"].pop("net")
        for i in range(len(self.mlp)):
            xx = self.mlp[i](xx)
            xx = F.relu(xx)
            xx = self.dropout(xx)
        return xx
