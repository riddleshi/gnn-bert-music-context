"""PyTorch dataset over the cached synthetic (or real) corpus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.bert_encoder import Vocab
from src.config import ROOT, load_config
from src.synthetic import load_cached_corpus


@dataclass
class Batch:
    input_ids: torch.Tensor
    x: torch.Tensor
    adj: torch.Tensor
    mask: torch.Tensor
    tags: torch.Tensor
    genre: torch.Tensor
    valence: torch.Tensor
    arousal: torch.Tensor
    mel: torch.Tensor
    ids: list[str]
    captions: list[str]


class MusicContextDataset(Dataset):
    def __init__(
        self,
        tracks: list[dict[str, Any]],
        vocab: Vocab,
        max_len: int,
        split_ids: Sequence[str] | None = None,
        graph: str = "segment",
    ):
        idset = set(split_ids) if split_ids is not None else None
        self.tracks = [t for t in tracks if idset is None or t["id"] in idset]
        self.vocab = vocab
        self.max_len = max_len
        self.graph = graph

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        t = self.tracks[idx]
        if self.graph == "chord":
            x, adj = t["chord_x"], t["chord_adj"]
        else:
            x, adj = t["seg_x"], t["seg_adj"]
        return {
            "id": t["id"],
            "caption": t["caption"],
            "ids": self.vocab.encode(t["caption"], self.max_len),
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32)),
            "adj": torch.from_numpy(np.asarray(adj, dtype=np.float32)),
            "tags": torch.from_numpy(np.asarray(t["tags"], dtype=np.float32)),
            "genre": int(t["genre_idx"]),
            "valence": float(t["valence"]),
            "arousal": float(t["arousal"]),
            "mel": torch.from_numpy(np.asarray(t["mel_small"], dtype=np.float32)),
        }


def collate(samples: list[dict[str, Any]]) -> Batch:
    n_max = max(s["x"].shape[0] for s in samples)
    f = samples[0]["x"].shape[1]
    b = len(samples)
    x = torch.zeros(b, n_max, f)
    adj = torch.zeros(b, n_max, n_max)
    mask = torch.zeros(b, n_max)
    for i, s in enumerate(samples):
        n = s["x"].shape[0]
        x[i, :n] = s["x"]
        adj[i, :n, :n] = s["adj"]
        mask[i, :n] = 1.0
    return Batch(
        input_ids=torch.from_numpy(np.stack([s["ids"] for s in samples])),
        x=x,
        adj=adj,
        mask=mask,
        tags=torch.stack([s["tags"] for s in samples]),
        genre=torch.tensor([s["genre"] for s in samples], dtype=torch.long),
        valence=torch.tensor([s["valence"] for s in samples], dtype=torch.float32),
        arousal=torch.tensor([s["arousal"] for s in samples], dtype=torch.float32),
        mel=torch.stack([s["mel"] for s in samples]),
        ids=[s["id"] for s in samples],
        captions=[s["caption"] for s in samples],
    )


def make_loaders(
    vocab: Vocab,
    graph: str = "segment",
    cfg: dict[str, Any] | None = None,
) -> dict[str, DataLoader]:
    cfg = cfg or load_config()
    tracks = load_cached_corpus(cfg)
    splits = __import__("json").loads((ROOT / cfg["paths"]["splits"] / "splits.json").read_text())
    loaders = {}
    for name in ("train", "val", "test"):
        ds = MusicContextDataset(tracks, vocab, int(cfg["max_text_len"]), splits[name], graph=graph)
        loaders[name] = DataLoader(
            ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=(name == "train"),
            collate_fn=collate,
            drop_last=False,
        )
    loaders["_tracks"] = tracks
    loaders["_splits"] = splits
    return loaders
