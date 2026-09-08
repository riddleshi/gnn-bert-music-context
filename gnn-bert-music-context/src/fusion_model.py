"""Task 3: cross-attention GNN–BERT fusion and early-concat ablation.

    A = softmax(Q K^T / √d),   Q = g W_Q,   K = H_text W_K
    z = CONCAT(g, A H_text)
    ŷ = σ(W z)

Multi-task loss:

    L = L_tags + α ||v − v̂||² + β ||a − â||²
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.bert_encoder import TinyMusicBERT
from src.gnn_model import GNNEncoder


class CrossAttentionFusion(nn.Module):
    def __init__(self, graph_dim: int, text_dim: int, n_tags: int, dropout: float = 0.1):
        super().__init__()
        d = graph_dim
        self.wq = nn.Linear(graph_dim, d)
        self.wk = nn.Linear(text_dim, d)
        self.wv = nn.Linear(text_dim, d)
        self.scale = math.sqrt(d)
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_dim + d, graph_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tag_head = nn.Linear(graph_dim, n_tags)
        self.valence_head = nn.Linear(graph_dim, 1)
        self.arousal_head = nn.Linear(graph_dim, 1)

    def fuse(self, g: torch.Tensor, h_text: torch.Tensor, text_pad: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # g: (B, d), h_text: (B, L, d_t)
        q = self.wq(g).unsqueeze(1)  # (B, 1, d)
        k = self.wk(h_text)  # (B, L, d)
        v = self.wv(h_text)
        attn = torch.matmul(q, k.transpose(1, 2)) / self.scale  # (B, 1, L)
        if text_pad is not None:
            attn = attn.masked_fill(text_pad.unsqueeze(1), -1e9)
        w = torch.softmax(attn, dim=-1)
        ctx = torch.matmul(w, v).squeeze(1)
        z = self.out(torch.cat([g, ctx], dim=-1))
        return z, w.squeeze(1)

    def forward(self, g, h_text, text_pad=None):
        z, attn = self.fuse(g, h_text, text_pad)
        tags = self.tag_head(z)
        val = self.valence_head(z).squeeze(-1)
        aro = self.arousal_head(z).squeeze(-1)
        return {"z": z, "tag_logits": tags, "valence": val, "arousal": aro, "attn": attn}


class EarlyConcatFusion(nn.Module):
    def __init__(self, graph_dim: int, text_dim: int, n_tags: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(graph_dim + text_dim, graph_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tag_head = nn.Linear(graph_dim, n_tags)
        self.valence_head = nn.Linear(graph_dim, 1)
        self.arousal_head = nn.Linear(graph_dim, 1)

    def forward(self, g, h_text, text_pad=None):
        t = h_text[:, 0]
        z = self.mlp(torch.cat([g, t], dim=-1))
        return {
            "z": z,
            "tag_logits": self.tag_head(z),
            "valence": self.valence_head(z).squeeze(-1),
            "arousal": self.arousal_head(z).squeeze(-1),
            "attn": None,
        }


class GNNbertFusion(nn.Module):
    def __init__(
        self,
        gnn: GNNEncoder,
        bert: TinyMusicBERT,
        n_tags: int,
        mode: str = "cross_attention",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gnn = gnn
        self.bert = bert
        hidden = gnn.input.out_features
        if mode == "concat":
            self.fusion = EarlyConcatFusion(hidden, bert.hidden, n_tags, dropout)
        else:
            self.fusion = CrossAttentionFusion(hidden, bert.hidden, n_tags, dropout)
        self.mode = mode

    def forward(self, x, adj, mask, input_ids):
        _, g = self.gnn(x, adj, mask)
        h_text, _ = self.bert(input_ids)
        pad = input_ids.eq(0)
        return self.fusion(g, h_text, pad)
