"""GraphSAGE and GAT encoders with mean-pool readout (Task 2).

GraphSAGE update (course spec):

    h_i^{(l+1)} = σ( W^{(l)} · CONCAT( h_i^{(l)}, MEAN_{j ∈ N(i)} h_j^{(l)} ) )

Implemented in dense batched form: adjacency is already symmetrically
normalized with self-loops, so `adj @ h` is a mean-like neighborhood aggregate.
PyTorch Geometric is optional; this module has no PyG dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: (B, N, F), adj: (B, N, N)
        neigh = torch.bmm(adj, h)
        out = torch.cat([h, neigh], dim=-1)
        return F.relu(self.lin(self.dropout(out)))


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        assert out_dim % heads == 0
        self.head_dim = out_dim // heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.zeros(heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.attn)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        b, n, _ = h.shape
        wh = self.W(h).view(b, n, self.heads, self.head_dim)
        # Pairwise attention scores.
        left = wh.unsqueeze(2).expand(b, n, n, self.heads, self.head_dim)
        right = wh.unsqueeze(1).expand(b, n, n, self.heads, self.head_dim)
        cat = torch.cat([left, right], dim=-1)
        e = self.leaky((cat * self.attn.view(1, 1, 1, self.heads, -1)).sum(-1))
        mask = adj > 0
        e = e.masked_fill(~mask.unsqueeze(-1), -1e9)
        alpha = torch.softmax(e, dim=2)
        alpha = self.dropout(alpha)
        # (B, N, N, H) x (B, N, H, D) -> (B, N, H, D)
        out = torch.einsum("bnih,bjhd->bnhd", alpha, wh)
        return F.elu(out.reshape(b, n, self.out_dim))


class GNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        layers: int = 2,
        kind: str = "graphsage",
        heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.kind = kind
        self.input = nn.Linear(in_dim, hidden)
        mods = []
        for _ in range(layers):
            if kind == "gat":
                mods.append(GATLayer(hidden, hidden, heads=heads, dropout=dropout))
            else:
                mods.append(GraphSAGELayer(hidden, hidden, dropout=dropout))
        self.layers = nn.ModuleList(mods)
        self.dropout = nn.Dropout(dropout)

    def node_embed(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = F.relu(self.input(x))
        for layer in self.layers:
            h = layer(h, adj)
            h = self.dropout(h)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        return h

    def readout(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Mean pooling: g = (1/|V|) Σ_i h_i^{(L)}."""
        if mask is None:
            return h.mean(dim=1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (h * mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.node_embed(x, adj, mask)
        g = self.readout(h, mask)
        return h, g


class GNNClassifier(nn.Module):
    def __init__(self, encoder: GNNEncoder, n_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.input.out_features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_labels))

    def forward(self, x, adj, mask=None):
        _, g = self.encoder(x, adj, mask)
        return self.head(g)
