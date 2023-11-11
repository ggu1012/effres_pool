import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
DEPRECATED!

Too much memory usage for generating hypergraph laplacian
--> Mainly being a problem for backward pass

"""



class HGNN(nn.Module):
    def __init__(self, Hs, ASMs, in_dim, hidden_dims, out_dim, device='cpu'):
        super().__init__()

        assert len(hidden_dims) % 2 == 0
        assert len(Hs) == len(ASMs) + 1
        assert len(Hs) * 2 - 2 == len(hidden_dims)

        self.pool_level = int(len(hidden_dims) / 2)
        coarsest_hidden_dim_idx = int(len(hidden_dims) / 2) - 1

        self.W = []
        self.W.append(nn.Linear(in_dim, hidden_dims[0]).to(device))

        for i in range(int(len(hidden_dims) / 2) - 1):
            self.W.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]).to(device))

        self.W.append(
            nn.Linear(
                hidden_dims[coarsest_hidden_dim_idx],
                hidden_dims[coarsest_hidden_dim_idx + 1],
            ).to(device)
        )

        # Uplift and MLP with skip connection
        for i in range(coarsest_hidden_dim_idx + 1, len(hidden_dims)-1):
            self.W.append(
                nn.Linear(
                    hidden_dims[i] + hidden_dims[len(hidden_dims) - 1 - i],
                    hidden_dims[i + 1],
                ).to(device)
            )

        self.W.append(nn.Linear(hidden_dims[0] + hidden_dims[-1], out_dim).to(device))
        self.ASMs = [x.to(device) for x in ASMs]
        self.P = []
        for ASM in self.ASMs:
            # ASM.shape == (cluster, num_nodes)
            D = dglsp.diag(1 / ASM.sum(dim=1))
            Uplift = D @ ASM
            self.P.append(Uplift.to(device))

        self.dropout = nn.Dropout(0.5).to(device)

        ###########################################################
        # (HIGHLIGHT) Compute the Laplacian with Sparse Matrix API
        ###########################################################

        self.Ls = []
        for H in Hs:
            H = H.to(device)
            # Compute node degree.
            d_V = H.sum(1).to(device)
            # Compute edge degree.
            d_E = H.sum(0).to(device)
            # Compute the inverse of the square root of the diagonal D_v.
            D_v_invsqrt = dglsp.diag(d_V**-0.5).to(device)
            # Compute the inverse of the diagonal D_e.
            D_e_inv = dglsp.diag(d_E**-1).to(device)
            # In our example, B is an identity matrix.
            n_edges = d_E.shape[0]
            B = dglsp.identity((n_edges, n_edges)).to(device)
            # Compute Laplacian from the equation above.
            x = D_v_invsqrt @ H
            y = B @ D_e_inv
            z = H.transpose() @ D_v_invsqrt
            L = x @ (y @ z)
            self.Ls.append(L)

        print(f"Ls shape: {[x.shape for x in self.Ls]}")
        print(f"P shape: {[x.shape for x in self.P]}")

    def forward(self, X):
        hidden = []

        # Goes down
        #   MLP
        # o ----
        #        \  Pool
        #         \
        #          o
        for i in range(self.pool_level):
            X = self.core_forward(i, X, rev=False)
            X = F.relu(X)
            hidden.append(X)
            X = self.pool(X, i)

        X = self.core_forward(self.pool_level, X)
        X = F.relu(X)

        # Goes up
        #              MLP
        #             ---- o
        #            /
        #   Uplift  /
        #          o

        for i in range(self.pool_level, 2*self.pool_level - 1):
            X = self.unravel(X, i)
            X = torch.hstack([X, hidden[self.pool_level - i - 1]]) # -1, -2, ...
            X = self.core_forward(i + 1, X)
            X = F.relu(X)

        # Final output dim
        X = self.unravel(X, 2*self.pool_level - 1)
        X = torch.hstack([X, hidden[0]])
        X = self.core_forward(2 * self.pool_level, X)

        return X

    def core_forward(self, i, X, rev=False):
        X = self.dropout(X)
        X = self.W[i](X)
        if i <= self.pool_level:
            X = self.Ls[i] @ X
        else:
            X = self.Ls[self.pool_level - i - 1] @ X
        return X

    def pool(self, X, i):
        """
        H: Incidence matrix (dgl.sparse)
        X: Node feature
        i: coarse level
        """
        return self.P[i] @ X

    def unravel(self, X, i):
        return self.ASMs[self.pool_level - i - 1].T @ X
