import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F


class HGNN(nn.Module):
    def __init__(self, Hs, ASMs, in_dim, hidden_dims, out_dim):
        super().__init__()

        assert len(hidden_dims) % 2 == 0
        assert len(Hs) == len(ASMs) + 1
        assert len(Hs) * 2 - 2 == len(hidden_dims)

        self.pool_level = len(hidden_dims) / 2
        coarsest_hidden_dim_idx = len(hidden_dims) / 2

        self.W = []
        self.W.append(nn.Linear(in_dim, hidden_dims[0]))

        for i in range(int(len(hidden_dims) / 2) - 1):
            self.W.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        self.W.append(
            nn.Linear(
                hidden_dims[coarsest_hidden_dim_idx],
                hidden_dims[coarsest_hidden_dim_idx + 1],
            )
        )

        # skip connection
        for i in range(coarsest_hidden_dim_idx + 1, len(hidden_dims)):
            self.W.append(
                nn.Linear(
                    hidden_dims[i] + hidden_dims[len(hidden_dims) - 1 - i],
                    hidden_dims[i + 1],
                )
            )

        self.W.append(nn.Linear(hidden_dims[-1], out_dim))

        self.ASMs = ASMs
        self.P = []
        for ASM in self.ASMs:
            # ASM.shape == (cluster, num_nodes)
            D = dglsp.diag(1 / ASM.sum(dim=1))
            self.P.append(D @ ASM)

        self.dropout = nn.Dropout(0.5)

        ###########################################################
        # (HIGHLIGHT) Compute the Laplacian with Sparse Matrix API
        ###########################################################

        self.Ls = []
        for H in Hs:
            # Compute node degree.
            d_V = H.sum(1)
            # Compute edge degree.
            d_E = H.sum(0)
            # Compute the inverse of the square root of the diagonal D_v.
            D_v_invsqrt = dglsp.diag(d_V**-0.5)
            # Compute the inverse of the diagonal D_e.
            D_e_inv = dglsp.diag(d_E**-1)
            # In our example, B is an identity matrix.
            n_edges = d_E.shape[0]
            B = dglsp.identity((n_edges, n_edges))
            # Compute Laplacian from the equation above.
            self.Ls.append(D_v_invsqrt @ H @ B @ D_e_inv @ H.T @ D_v_invsqrt)

    def forward(self, X):
        hidden = []

        # Goes down
        # o ----
        #        \
        #         \
        #          o
        for i in range(self.pool_level):
            X = self.core_forward(i, X)
            X = F.relu(X)
            hidden.append(X)
            X = self.pool(X, i)

        X = self.core_forward(self.pool_level, X)
        X = F.relu(X)

        # Goes up
        for i in range(self.pool_level - 1):
            X = self.unravel(X, i)
            X = torch.concat([X, hidden[self.pool_level - i - 1]])
            X = self.core_forward(self.pool_level + i, X)
            X = F.relu(X)

        # Final output dim
        X = self.core_forward(2 * self.pool_level, X)

    def core_forward(self, i, X):
        X = self.dropout(X)
        X = self.W[i](X)
        if i <= self.pool_level:
            X = self.L[i] @ X
        else:
            X = self.L[self.pool_level - i] @ X
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

