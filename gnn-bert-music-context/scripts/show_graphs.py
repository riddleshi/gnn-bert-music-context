#!/usr/bin/env python3
"""Load preprocessed .pt graphs with torch.load and draw them.

Example (assignment-style):

    import torch
    graph = torch.load("data/processed/graphs/track_000_chord.pt", weights_only=False)
    print(graph.keys())
    print(graph["node_labels"])

Usage:
    PYTHONPATH=. python scripts/show_graphs.py
    PYTHONPATH=. python scripts/show_graphs.py --file data/processed/graphs/track_000_segment.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from src.config import ROOT
from src.load_graph import draw_pt_graph, load_pt_graph


def inspect(path: Path) -> None:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    graph = load_pt_graph(path)
    print(f"\n=== {path.relative_to(ROOT)} ===")
    print("keys:", list(graph.keys()))
    print("kind:", graph.get("kind"))
    print("node_labels:", graph["node_labels"])
    print("x shape:", tuple(graph["x"].shape), "adj:", tuple(graph["adj"].shape))
    print("edges:", int(graph["edge_index"].shape[1]))
    stem = path.stem
    png = ROOT / "results" / "plots" / "graphs" / f"{stem}.png"
    draw_pt_graph(graph, png, title=f"{stem}  ({graph.get('kind')})")
    print("wrote", png.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None, help="Single .pt path; default = all samples")
    args = parser.parse_args()
    if args.file:
        inspect(Path(args.file))
        return
    folder = ROOT / "data" / "processed" / "graphs"
    files = sorted(folder.glob("*.pt"))
    if not files:
        raise SystemExit(f"no .pt graphs in {folder}")
    for path in files:
        inspect(path)
    print(f"\n{len(files)} graphs loaded with torch.load and drawn to results/plots/graphs/")


if __name__ == "__main__":
    main()
