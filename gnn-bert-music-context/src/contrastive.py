"""Task 4: dual-encoder InfoNCE alignment of graphs and captions.

    L_NCE = -log  exp(sim(g_i, t_i)/τ) / Σ_j exp(sim(g_i, t_j)/τ)
    sim(u, v) = u^T v / (||u|| ||v||)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bert_encoder import TinyMusicBERT
from src.gnn_model import GNNEncoder


def info_nce(graph_z: torch.Tensor, text_z: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    g = F.normalize(graph_z, dim=-1)
    t = F.normalize(text_z, dim=-1)
    logits = g @ t.T / temperature
    targets = torch.arange(g.size(0), device=g.device)
    loss_g = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_g + loss_t)


class DualEncoder(nn.Module):
    def __init__(self, gnn: GNNEncoder, bert: TinyMusicBERT, proj_dim: int = 64):
        super().__init__()
        self.gnn = gnn
        self.bert = bert
        hidden_g = gnn.input.out_features
        self.graph_proj = nn.Sequential(nn.Linear(hidden_g, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.text_proj = nn.Sequential(nn.Linear(bert.hidden, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))

    def encode_graph(self, x, adj, mask=None):
        _, g = self.gnn(x, adj, mask)
        return self.graph_proj(g)

    def encode_text(self, input_ids):
        _, cls = self.bert(input_ids)
        return self.text_proj(cls)

    def forward(self, x, adj, mask, input_ids, temperature: float = 0.07):
        g = self.encode_graph(x, adj, mask)
        t = self.encode_text(input_ids)
        loss = info_nce(g, t, temperature)
        return {"loss": loss, "graph_z": g, "text_z": t}


@torch.no_grad()
def retrieval_metrics(
    graph_z: torch.Tensor,
    text_z: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    g = F.normalize(graph_z, dim=-1)
    t = F.normalize(text_z, dim=-1)
    sim = g @ t.T
    n = sim.size(0)
    out: dict[str, float] = {}
    for direction, matrix in (("caption_to_audio", sim.T), ("audio_to_caption", sim)):
        ranked = matrix.argsort(dim=1, descending=True)
        targets = torch.arange(n, device=sim.device).unsqueeze(1)
        hits = ranked == targets
        for k in ks:
            k_use = min(k, n)
            out[f"{direction}_R@{k}"] = float(hits[:, :k_use].any(dim=1).float().mean())
    return out
