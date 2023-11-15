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
    return {"cat_feat": torch.hstack([edges.src["x_"], edges.dst["x_"]])}


class EXGNN(nn.Module):
    def __init__(
        self, gnn_dims, mlp_dims, x_net_dim, tail_dims=-1, pool_level=2, device="cpu"
    ):
        super().__init__()

        assert len(gnn_dims) == 2 * (pool_level + 1)
        if isinstance(tail_dims, list):
            assert tail_dims[0] == gnn_dims[-1]
            assert mlp_dims[0] == tail_dims[-1] + x_net_dim
        else:
            assert mlp_dims[0] == gnn_dims[-1] + x_net_dim

        self.act = nn.Tanh()
        self.pool_level = pool_level
        floor_step = [
            # dglnn.GATConv(gnn_dims[i], gnn_dims[i + 1], num_heads=1, feat_drop=0.5).to(
            #     device
            # )
            dglnn.SAGEConv(
                gnn_dims[i], gnn_dims[i + 1], "mean", feat_drop=0.5, activation=self.act
            ).to(device)
            for i in range(self.pool_level + 1)
        ] + [
            # dglnn.GATConv(
            #     gnn_dims[i] + gnn_dims[2 * self.pool_level + 1 - i],
            #     gnn_dims[i + 1],
            #     num_heads=1,
            #     feat_drop=0.5,
            # ).to(device)
            dglnn.SAGEConv(
                gnn_dims[i] + gnn_dims[2 * self.pool_level + 1 - i],
                gnn_dims[i + 1],
                "mean",
                feat_drop=0.5,
                activation=self.act,
            ).to(device)
            for i in range(self.pool_level + 1, 2 * self.pool_level + 1)
        ]

        self.floor_step = nn.ModuleList(floor_step)

        if isinstance(tail_dims, list):
            self.tail_gnn = nn.ModuleList(
                [
                    dglnn.SAGEConv(
                        tail_dims[i],
                        tail_dims[i + 1],
                        "mean",
                        feat_drop=0.5,
                        activation=self.act,
                    ).to(device)
                    for i in range(len(tail_dims) - 1)
                ]
            )
            mlp = [nn.Linear(tail_dims[-1] + x_net_dim, mlp_dims[1]).to(device)]
        else:
            mlp = [nn.Linear(gnn_dims[-1] + x_net_dim, mlp_dims[1]).to(device)]
        mlp += [
            nn.Linear(mlp_dims[i], mlp_dims[i + 1]).to(device)
            for i in range(1, len(mlp_dims) - 1)
        ]
        self.mlp = nn.ModuleList(mlp)
        self.dropout = nn.Dropout(0.5).to(device)

    def forward(self, gr):
        # Goes down
        #   GNN
        # o ---- o
        # x     x_\  Pool
        #          \
        #           o
        #           x

        layer_subg = [
            dgl.edge_type_subgraph(gr, [(f"lv{i}", "to", f"lv{i}")])
            for i in range(self.pool_level + 1)
        ]

        for i in range(self.pool_level):
            sg = layer_subg[i]
            sg.ndata["x_"] = self.floor_step[i](sg, sg.ndata["x"])

            sg = dgl.edge_type_subgraph(gr, [(f"lv{i}", "downwards", f"lv{i+1}")])
            sg.update_all(fn.copy_u("x_", "m"), fn.mean("m", "x"))

        # Bottom
        # o --- o
        # x     x_
        sg = layer_subg[self.pool_level]
        sg.ndata["x_"] = self.floor_step[self.pool_level](sg, sg.ndata["x"])

        # Goes up
        #    concat      GNN
        #    ----->   o ---- o
        #            /x__    x_
        #   Uplift  /
        #          o
        #          x_ (pop)

        pl = self.pool_level
        for i in range(pl):
            rev_i = pl - i - 1
            sg = dgl.edge_type_subgraph(gr, [(f"lv{rev_i+1}", "upwards", f"lv{rev_i}")])
            sg.update_all(scatter_cat_up, fn.sum("cat_feat", "x__"))
            sg = layer_subg[rev_i]
            _ = sg.ndata.pop("x_")
            sg.ndata["x_"] = self.floor_step[pl + i + 1](sg, sg.ndata["x__"])

        # Tail GNN
        sgr = layer_subg[0]
        xx = sgr.ndata["x_"]
        if hasattr(self, "tail_gnn"):
            for layer in self.tail_gnn:
                xx = layer(sgr, xx)

        sgr.ndata["x1"], sgr.ndata["x2"] = torch.hsplit(xx, 2)

        # message passing to net node
        sg = dgl.edge_type_subgraph(gr, ["conn"])
        sg.update_all(fn.copy_u("x1", "m"), fn.max("m", "y_max"))
        sg.update_all(fn.copy_u("x2", "m"), fn.max("m", "y_min"))
        xx = torch.hstack(
            [
                sg.ndata["y_max"]["net"],
                sg.ndata["y_min"]["net"],
                sg.ndata["x_net"]["net"],
            ]
        )

        for i in range(len(self.mlp) - 1):
            xx = self.mlp[i](xx)
            xx = self.act(xx)
            xx = self.dropout(xx)
        xx = self.mlp[-1](xx)
        # xx = self.final_act(xx)

        return xx
