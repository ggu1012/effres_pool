import dgl.sparse as dglsp
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn.pytorch as dglnn

from utils.hypergraph_conversion import ExpanderCliqueW
"""
Utilize 3-cycle uniform expander graph converted from hypergraph

"""

# dgl udf
def scatter_cat_up(edges):
    return {'cat_feat': torch.hstack([edges.src['x'], edges.dst['x']])}

class EXGNN(nn.Module):
    def __init__(self, net2nodes, ASMs, dims, device='cpu'):
        super().__init__()

        assert len(dims) % 2 == 0
        assert len(net2nodes) == len(ASMs) + 1
        assert len(net2nodes) * 2 == len(dims)

        self.pool_level = len(net2nodes) - 1

        self.floor_step = [
            dglnn.SAGEConv(dims[i], dims[i+1], 'mean', feat_drop=0.5).to(device)
            for i in range(self.pool_level + 1)
        ] + \
        [
            # dglnn.SAGEConv(dims[i], dims[i+1], 'mean', feat_drop=0.5).to(device)
            dglnn.SAGEConv(dims[i] + dims[2 * self.pool_level + 1 - i], dims[i+1], 'mean', feat_drop=0.5).to(device)
            for i in range(self.pool_level + 1, 2 * self.pool_level + 1)
        ]

        print(self.floor_step)

        self.mlp = nn.Linear(dims[-1], 1).to(device)
        self.act = nn.Softmax(dim=0)
        

        self.dropout = nn.Dropout(0.5).to(device)

        """
        Build uniform 3-cycle expander graph from net2node list (incidence matrix)

        """
        data_dict = {}
        for lv, n2n in enumerate(net2nodes):
            key = (f'lv{lv}', 'to', f'lv{lv}')
            adj = ExpanderCliqueW(hinc=n2n, expander_sz=3)
            indptr = torch.tensor(adj.indptr, dtype=int)
            indices = torch.tensor(adj.indices, dtype=int)
            data_dict[key] = ('csr', (indptr, indices, []))
        
        for lv in range(len(net2nodes) - 1):
            data_dict[(f'lv{lv}', 'downwards', f'lv{lv+1}')] = \
                  ('csr', (ASMs[lv].T.csr()))
            data_dict[(f'lv{lv+1}', 'upwards', f'lv{lv}')] = \
                  ('csr', (ASMs[lv].csr()))
            
        self.gr = dgl.heterograph(data_dict).to(device)
        

    def forward(self, X):
        gr = self.gr
        gr.ndata['x'] = {'lv0': X}

        # Goes down
        #   GNN
        # o ----
        #        \  Pool
        #         \
        #          o

        for i in range(self.pool_level):
            sg = dgl.edge_type_subgraph(self.gr, [(f'lv{i}', 'to', f'lv{i}')])
            xx = gr.ndata['x'].pop(f'lv{i}')
            xx = self.floor_step[i](sg, xx)
            gr.ndata['x'] = {f'lv{i}': xx }

            sg = dgl.edge_type_subgraph(self.gr, [(f'lv{i}', 'downwards', f'lv{i+1}')])
            sg.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'x'))
            xx = gr.ndata['x'].pop(f'lv{i+1}')
            gr.ndata['x'][f'lv{i+1}'] = F.relu(xx)

        # Bottom
        sg = dgl.edge_type_subgraph(self.gr, [(f'lv{self.pool_level}', 'to', f'lv{self.pool_level}')])
        xx = gr.ndata['x'].pop(f'lv{self.pool_level}')
        xx = self.floor_step[self.pool_level](sg, xx)
        gr.ndata['x'][f'lv{self.pool_level}'] = F.relu(xx)

        # Goes up
        #              GNN
        #             ---- o
        #            /
        #   Uplift  /
        #          o

        pl = self.pool_level
        for i in range(pl):
            rev_i = pl - i - 1
            sg = dgl.edge_type_subgraph(self.gr, [(f'lv{rev_i+1}', 'upwards', f'lv{rev_i}')])
            sg.update_all(scatter_cat_up, fn.sum('cat_feat', 'x'))
            print(gr.ndata['x'][f'lv{rev_i}'].shape)
            sg = dgl.edge_type_subgraph(self.gr, [(f'lv{rev_i}', 'to', f'lv{rev_i}')])
            xx = sg.ndata.pop('x')
            xx = self.floor_step[pl + i + 1](sg, xx)
            xx = F.relu(xx)
            sg.ndata['x'] = xx

        xx = gr.ndata['x'].pop(f'lv0')
        xx = self.mlp(xx)
        
        return self.act(xx)
    
    