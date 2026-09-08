"""Chord-transition and segment-similarity graphs.

Chord-transition graph: nodes = unique chords; edges = observed transitions
weighted by count.

Segment graph: nodes = time segments; edges = temporal adjacency plus
cosine similarity of MFCC/chroma > τ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from src.audio_features import (
    AudioBundle,
    beat_synchronous_slices,
    pool_segment,
    segment_slices,
)
from src.config import load_config
from src.taxonomy import CHORD_NAMES

PITCH_CLASSES = np.array("C C# D D# E F F# G G# A A# B".split())


def _major_template() -> np.ndarray:
    t = np.zeros(12, dtype=np.float32)
    t[[0, 4, 7]] = [1.0, 0.8, 0.9]
    return t


def _minor_template() -> np.ndarray:
    t = np.zeros(12, dtype=np.float32)
    t[[0, 3, 7]] = [1.0, 0.8, 0.9]
    return t


def chord_templates() -> np.ndarray:
    """24 templates (12 maj + 12 min), shape (24, 12)."""
    maj = _major_template()
    minor = _minor_template()
    rows = []
    for shift in range(12):
        rows.append(np.roll(maj, shift))
    for shift in range(12):
        rows.append(np.roll(minor, shift))
    T = np.stack(rows, axis=0)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    return T.astype(np.float32)


TEMPLATES = chord_templates()


def estimate_chords(chroma: np.ndarray) -> np.ndarray:
    """Frame-wise chord index in [0, 23] via cosine match to templates."""
    C = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8)
    scores = TEMPLATES @ C
    return scores.argmax(axis=0)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


@dataclass
class MusicGraph:
    x: np.ndarray  # (N, F)
    adj: np.ndarray  # (N, N) weighted, with self-loops
    edge_index: np.ndarray  # (2, E)
    edge_weight: np.ndarray  # (E,)
    node_labels: list[str]
    kind: str
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    def to_networkx(self) -> nx.Graph:
        g = nx.Graph()
        for i, lab in enumerate(self.node_labels):
            g.add_node(i, label=lab)
        for (s, t), w in zip(self.edge_index.T, self.edge_weight):
            if int(s) == int(t):
                continue
            if g.has_edge(int(s), int(t)):
                g[int(s)][int(t)]["weight"] = max(g[int(s)][int(t)]["weight"], float(w))
            else:
                g.add_edge(int(s), int(t), weight=float(w))
        return g

    def layout(self) -> list[dict[str, float]]:
        g = self.to_networkx()
        if g.number_of_nodes() == 0:
            return []
        pos = nx.spring_layout(g, seed=0, weight="weight", iterations=80)
        out = []
        for i in range(self.num_nodes):
            x, y = pos.get(i, (0.0, 0.0))
            out.append({"x": float(x), "y": float(y)})
        return out

    def to_json(self) -> dict[str, Any]:
        layout = self.layout()
        nodes = []
        for i, lab in enumerate(self.node_labels):
            nodes.append(
                {
                    "id": i,
                    "label": lab,
                    "x": layout[i]["x"] if layout else 0.0,
                    "y": layout[i]["y"] if layout else 0.0,
                    "energy": float(np.linalg.norm(self.x[i])),
                }
            )
        edges = []
        seen = set()
        for (s, t), w in zip(self.edge_index.T, self.edge_weight):
            a, b = int(s), int(t)
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            edges.append({"source": a, "target": b, "weight": float(w)})
        return {"kind": self.kind, "nodes": nodes, "edges": edges, **self.extras}

    def to_torch(self) -> dict[str, Any]:
        import torch

        return {
            "x": torch.from_numpy(self.x),
            "adj": torch.from_numpy(self.adj),
            "edge_index": torch.from_numpy(self.edge_index),
            "edge_weight": torch.from_numpy(self.edge_weight),
            "node_labels": self.node_labels,
            "kind": self.kind,
        }


def _sym_norm(adj: np.ndarray) -> np.ndarray:
    adj = adj.copy()
    np.fill_diagonal(adj, adj.diagonal() + 1.0)
    deg = adj.sum(axis=1)
    d_inv = np.power(np.maximum(deg, 1e-8), -0.5)
    D = np.diag(d_inv)
    return (D @ adj @ D).astype(np.float32)


def _from_weighted_adj(x: np.ndarray, w: np.ndarray, labels: list[str], kind: str, **extras) -> MusicGraph:
    n = x.shape[0]
    adj = _sym_norm(w)
    src, dst = np.where(w > 0)
    mask = src != dst
    if not mask.any():
        # isolated nodes: keep self-loops only
        src = np.arange(n)
        dst = np.arange(n)
        weights = np.ones(n, dtype=np.float32)
    else:
        src, dst = src[mask], dst[mask]
        weights = w[src, dst].astype(np.float32)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    return MusicGraph(
        x=x.astype(np.float32),
        adj=adj,
        edge_index=edge_index,
        edge_weight=weights,
        node_labels=labels,
        kind=kind,
        extras=extras,
    )


def build_segment_graph(bundle: AudioBundle, use_beats: bool = False) -> MusicGraph:
    cfg = load_config()
    hop = int(cfg["hop_length"])
    tau = float(cfg["similarity_tau"])
    n_frames = bundle.log_mel.shape[1]
    if use_beats and len(bundle.beat_frames) >= 4:
        slices = beat_synchronous_slices(bundle.beat_frames, n_frames, group=2)
    else:
        slices = segment_slices(n_frames, bundle.sr, hop, float(cfg["segment_seconds"]))

    feats = []
    chroma_mu = []
    mfcc_mu = []
    labels = []
    for i, (a, b) in enumerate(slices):
        feats.append(pool_segment(bundle, a, b))
        chroma_mu.append(bundle.chroma[:, a:b].mean(axis=1))
        mfcc_mu.append(bundle.mfcc[:, a:b].mean(axis=1))
        labels.append(f"seg{i}")

    x = np.stack(feats, axis=0)
    n = x.shape[0]
    w = np.zeros((n, n), dtype=np.float32)

    # Temporal adjacency (bidirectional next-segment edges).
    for i in range(n - 1):
        w[i, i + 1] = 1.0
        w[i + 1, i] = 1.0

    # Similarity edges from chroma+MFCC cosine.
    sim_src = np.concatenate([np.stack(chroma_mu), np.stack(mfcc_mu)], axis=1)
    sim_src = sim_src / (np.linalg.norm(sim_src, axis=1, keepdims=True) + 1e-8)
    sim = sim_src @ sim_src.T
    sim = np.clip(sim, 0, 1)
    np.fill_diagonal(sim, 0.0)
    w = np.maximum(w, (sim > tau).astype(np.float32) * sim)

    return _from_weighted_adj(
        x,
        w,
        labels,
        kind="segment",
        tau=tau,
        n_segments=n,
        slices=[(int(a), int(b)) for a, b in slices],
    )


def build_chord_graph(bundle: AudioBundle) -> MusicGraph:
    """Nodes are unique chords observed in the clip; edges are transitions."""
    hop = int(load_config()["hop_length"])
    n_frames = bundle.chroma.shape[1]
    slices = segment_slices(n_frames, bundle.sr, hop, float(load_config()["segment_seconds"]))
    chord_seq = []
    node_feat_acc: dict[int, list[np.ndarray]] = {}
    for a, b in slices:
        local = estimate_chords(bundle.chroma[:, a:b])
        # Majority chord in the window.
        counts = np.bincount(local, minlength=24)
        cid = int(counts.argmax())
        chord_seq.append(cid)
        node_feat_acc.setdefault(cid, []).append(pool_segment(bundle, a, b))

    if not chord_seq:
        chord_seq = [0]
        node_feat_acc[0] = [np.zeros(bundle.log_mel.shape[0] * 2 + 50, dtype=np.float32)]

    unique = sorted(node_feat_acc.keys())
    idx = {c: i for i, c in enumerate(unique)}
    n = len(unique)
    x = np.stack([np.mean(node_feat_acc[c], axis=0) for c in unique], axis=0)
    trans = np.zeros((n, n), dtype=np.float32)
    for u, v in zip(chord_seq[:-1], chord_seq[1:]):
        trans[idx[u], idx[v]] += 1.0
    # Make undirected for GraphSAGE mean-aggregation while keeping weights.
    w = trans + trans.T
    if w.sum() == 0:
        # Single-chord clip: connect the node to itself via a dummy neighbor clone.
        w[0, 0] = 1.0
    labels = [CHORD_NAMES[c] if c < len(CHORD_NAMES) else f"chord{c}" for c in unique]
    return _from_weighted_adj(
        x,
        w,
        labels,
        kind="chord",
        chord_sequence=[CHORD_NAMES[c] for c in chord_seq],
        n_unique=n,
    )


def build_graphs(bundle: AudioBundle) -> dict[str, MusicGraph]:
    return {
        "segment": build_segment_graph(bundle, use_beats=False),
        "chord": build_chord_graph(bundle),
    }


def graph_coherence(graph: MusicGraph, tau: float = 0.5) -> float:
    """Optional analysis: fraction of edges whose endpoints have cosine(h_i, h_j) > τ."""
    if graph.edge_index.size == 0:
        return 0.0
    x = graph.x / (np.linalg.norm(graph.x, axis=1, keepdims=True) + 1e-8)
    hits = 0
    n_e = 0
    seen = set()
    for s, t in graph.edge_index.T:
        a, b = int(s), int(t)
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        n_e += 1
        if float(np.dot(x[a], x[b])) > tau:
            hits += 1
    return hits / max(n_e, 1)
