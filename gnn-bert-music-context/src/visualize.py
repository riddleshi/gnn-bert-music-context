"""Publication-style plots for the report and the web demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
PALETTE = ["#c45c26", "#1f6f8b", "#2f4b3a", "#c9a227", "#6b3fa0", "#b33939", "#2c3e50", "#4a7c59"]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_f1_curves(history: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    mapping = [
        ("task1", "val_macro_f1", "BERT-only"),
        ("task2_tags", "val_macro_f1", "GNN-only"),
        ("task3_cross_attention", "val_macro_f1", "GNN–BERT (cross-attn)"),
        ("task3_concat", "val_macro_f1", "GNN–BERT (concat)"),
        ("cnn_tags", "val_macro_f1", "CNN mel-spec"),
    ]
    for key, field, label in mapping:
        rows = history.get(key) or []
        ys = [r.get(field) for r in rows if r.get(field) is not None]
        if ys:
            ax.plot(range(1, len(ys) + 1), ys, marker="o", label=label, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Tag understanding vs. training epoch")
    ax.legend(frameon=False, fontsize=11)
    ax.set_ylim(0, 1.02)
    _save(fig, path)


def save_ablation_bars(models: dict[str, Any], path: Path) -> None:
    names = ["random", "majority", "cnn_mel", "bert_only", "gnn_only", "fusion_concat", "fusion_cross_attention"]
    labels = ["Random", "Majority", "CNN", "BERT", "GNN", "Early concat", "Cross-attn"]
    f1 = [models.get(n, {}).get("macro_f1", 0.0) for n in names]
    auc = [models.get(n, {}).get("auc_pr", 0.0) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.38
    ax.bar(x - w / 2, f1, w, label="Macro-F1", color=PALETTE[0])
    ax.bar(x + w / 2, auc, w, label="AUC-PR", color=PALETTE[1])
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Ablation: tag understanding on the held-out split")
    ax.legend(frameon=False)
    _save(fig, path)


def save_tsne(z2: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    uniq = list(dict.fromkeys(labels))
    for i, lab in enumerate(uniq):
        m = np.array([l == lab for l in labels])
        ax.scatter(z2[m, 0], z2[m, 1], s=42, color=PALETTE[i % len(PALETTE)], label=lab, alpha=0.85, edgecolors="none")
    ax.legend(frameon=False, fontsize=9, markerscale=1.1, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    _save(fig, path)


def save_va_plane(v_t, a_t, v_p, a_p, labels: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    uniq = list(dict.fromkeys(labels))
    for i, lab in enumerate(uniq):
        m = np.array([l == lab for l in labels])
        ax.scatter(v_t[m], a_t[m], s=36, color=PALETTE[i % len(PALETTE)], label=lab, alpha=0.7)
        ax.scatter(v_p[m], a_p[m], s=22, facecolors="none", edgecolors=PALETTE[i % len(PALETTE)], alpha=0.9)
    ax.axhline(0.5, color="#999", lw=0.8)
    ax.axvline(0.5, color="#999", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title("DEAM-style emotion plane (fill = true, ring = predicted)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    _save(fig, path)


def save_retrieval_panel(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 1.1 * max(3, len(rows))))
    ax.axis("off")
    y = 1.0
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    dy = 1.0 / (len(rows) + 1)
    y = 1 - dy * 0.4
    ax.set_title("Caption → audio retrieval (top-3)", pad=12)
    for row in rows:
        q = row["query_caption"]
        ax.text(0.01, y, f"Q · {row['query_genre']}: {q[:110]}", fontsize=8, va="top", wrap=True)
        y -= dy * 0.45
        hits = ",  ".join(
            f"{h['title']} ({h['genre']}, {h['score']:.2f}{' ✓' if h['correct'] else ''})" for h in row["hits"]
        )
        ax.text(0.04, y, hits, fontsize=8, va="top", color="#333")
        y -= dy * 0.55
    _save(fig, path)
