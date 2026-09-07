"""Whitespace tokenizer with a corpus-built vocabulary (BERT stand-in).

Swap `TinyMusicBERT` for HuggingFace `bert-base-uncased` by setting
`model.hf_name` in config.yaml — the encoder interface stays the same:
CLS vector + full token hidden states.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
PAD, UNK, CLS, SEP = 0, 1, 2, 3


def tokenize(text: str) -> list[str]:
    text = text.lower().replace("–", "-").replace("—", "-")
    return re.findall(r"[a-z0-9']+|[ivx]+", text)


class Vocab:
    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}

    def __len__(self) -> int:
        return len(self.stoi)

    def encode(self, text: str, max_len: int) -> np.ndarray:
        tokens = tokenize(text)[: max_len - 2]
        ids = [CLS] + [self.stoi.get(t, UNK) for t in tokens] + [SEP]
        out = np.full(max_len, PAD, dtype=np.int64)
        out[: len(ids)] = np.array(ids, dtype=np.int64)[:max_len]
        return out

    def encode_batch(self, texts: Iterable[str], max_len: int) -> np.ndarray:
        return np.stack([self.encode(t, max_len) for t in texts], axis=0)

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(self.stoi, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(path.read_text()))


def build_vocab(captions: list[str], max_size: int = 2500, min_freq: int = 1) -> Vocab:
    counts = Counter()
    for cap in captions:
        counts.update(tokenize(cap))
        counts.update(tokenize(cap))  # slight boost to in-corpus words
    stoi = {s: i for i, s in enumerate(SPECIAL)}
    for tok, c in counts.most_common():
        if c < min_freq:
            continue
        if tok not in stoi:
            stoi[tok] = len(stoi)
        if len(stoi) >= max_size:
            break
    return Vocab(stoi)


class TinyMusicBERT(nn.Module):
    """2-layer Transformer encoder with a CLS token — Algorithm 1 backbone.

    Hidden size is intentionally small so CPU training finishes in minutes while
    the update equations match BERT-style contextualization.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden: int = 64,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 48,
    ):
        super().__init__()
        self.hidden = hidden
        self.tok = nn.Embedding(vocab_size, hidden, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.max_len = max_len
        self.register_buffer("pos_ids", torch.arange(max_len).unsqueeze(0), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (H_text [B, L, d], t_cls [B, d])."""
        b, l = input_ids.shape
        pos = self.pos_ids[:, :l].expand(b, l)
        x = self.tok(input_ids) + self.pos(pos)
        pad_mask = input_ids.eq(PAD)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        cls = h[:, 0]
        return h, cls


class BertTagHead(nn.Module):
    """Task 1: ŷ_k = σ(w_k^T t + b_k)."""

    def __init__(self, encoder: TinyMusicBERT, n_tags: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(encoder.hidden, n_tags))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, cls = self.encoder(input_ids)
        return self.head(cls)
