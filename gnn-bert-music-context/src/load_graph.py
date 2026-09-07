"""Load saved music graphs from .pt files (the assignment sample format).

    import torch
    graph = torch.load("data/processed/graphs/track_000_chord.pt", weights_only=False)
    print(graph.keys())
    print(graph["node_labels"])

`graph` is a dict:
  x            float tensor [N, F]  node features (mel/chroma/MFCC pooled)
  adj          float tensor [N, N]  symmetrically normalized adjacency
  edge_index   long tensor  [2, E]  COO edges
  edge_weight  float tensor [E]
  node_labels  list[str]            "seg0" / "Amin" / ...
  kind         "segment" | "chord"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch

from src.config import ROOT


def load_pt_graph(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    blob = torch.load(path, weights_only=False, map_location="cpu")
    if not isinstance(blob, dict) or "x" not in blob:
        raise ValueError(f"{path} is not a Cadence music graph .pt file")
    return blob


def graph_to_networkx(blob: dict[str, Any]) -> nx.Graph:
    g = nx.Graph()
    labels = list(blob["node_labels"])
    for i, lab in enumerate(labels):
        g.add_node(i, label=lab)
    edge_index = blob["edge_index"].cpu().numpy()
    weights = blob["edge_weight"].cpu().numpy()
    for (s, t), w in zip(edge_index.T, weights):
        a, b = int(s), int(t)
        if a == b:
            continue
        if g.has_edge(a, b):
            g[a][b]["weight"] = max(g[a][b]["weight"], float(w))
        else:
            g.add_edge(a, b, weight=float(w))
    return g


def graph_to_draw_json(blob: dict[str, Any]) -> dict[str, Any]:
    """JSON the lab UI GraphView expects, laid out from the .pt adjacency."""
    g = graph_to_networkx(blob)
    n = int(blob["x"].shape[0])
    if g.number_of_nodes() == 0:
        pos = {i: (0.0, 0.0) for i in range(n)}
    else:
        pos = nx.spring_layout(g, seed=0, weight="weight", iterations=80)
    labels = list(blob["node_labels"])
    x = blob["x"].cpu().numpy()
    nodes = []
    for i in range(n):
        px, py = pos.get(i, (0.0, 0.0))
        nodes.append(
            {
                "id": i,
                "label": labels[i] if i < len(labels) else f"n{i}",
                "x": float(px),
                "y": float(py),
                "energy": float(np.linalg.norm(x[i])),
            }
        )
    edges = []
    seen: set[tuple[int, int]] = set()
    edge_index = blob["edge_index"].cpu().numpy()
    weights = blob["edge_weight"].cpu().numpy()
    for (s, t), w in zip(edge_index.T, weights):
        a, b = int(s), int(t)
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        edges.append({"source": a, "target": b, "weight": float(w)})
    return {
        "kind": str(blob.get("kind", "segment")),
        "nodes": nodes,
        "edges": edges,
        "num_nodes": n,
        "feat_dim": int(blob["x"].shape[1]),
        "source": "torch.load",
    }


def draw_pt_graph(blob: dict[str, Any], out_path: str | Path | None = None, title: str | None = None):
    """Matplotlib drawing used by the notebook and scripts/show_graphs.py."""
    import matplotlib.pyplot as plt

    g = graph_to_networkx(blob)
    pos = nx.spring_layout(g, seed=0, weight="weight", iterations=80) if g.number_of_nodes() else {}
    labels = {i: lab for i, lab in enumerate(blob["node_labels"])}
    color = "#5ee0c0" if blob.get("kind") == "segment" else "#e8a05a"
    fig, ax = plt.subplots(figsize=(6.2, 5.2), facecolor="#141210")
    ax.set_facecolor("#141210")
    if g.number_of_edges():
        widths = [1.0 + 2.0 * g[u][v].get("weight", 1.0) for u, v in g.edges()]
        nx.draw_networkx_edges(g, pos, ax=ax, width=widths, edge_color="#9a8f84", alpha=0.85)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=color, node_size=720, edgecolors="#f4efe8")
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=8, font_color="#100e0c")
    ax.set_axis_off()
    ax.set_title(title or str(blob.get("kind", "graph")), color="#f4efe8")
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return str(out_path)
    return fig
