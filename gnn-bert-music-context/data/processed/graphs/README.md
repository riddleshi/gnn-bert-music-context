# Preprocessed graph samples

Assignment deliverable: at least 20 example `.pt` / `.json` graphs.

This folder has **48 files** from 12 clips (tracks 000–011):

| Pattern | Count | Contents |
|---|---:|---|
| `track_*_segment.json` / `.pt` | 24 | 1 s time-segment graph (temporal + chroma/MFCC similarity edges) |
| `track_*_chord.json` / `.pt` | 24 | Chord-transition graph (nodes = unique chords, edges = observed transitions) |

JSON is for inspection and the lab UI. `.pt` is a torch dict with `x`, `adj`, `edge_index`, `edge_weight`, `node_labels`, `kind`.

```python
import torch

graph = torch.load(
    "data/processed/graphs/track_000_chord.pt",
    weights_only=False,
)

print(graph.keys())
print(graph["node_labels"])
```

Draw every sample: `PYTHONPATH=. python scripts/show_graphs.py` (writes `results/plots/graphs/*.png`).

Built by `src/graph_builder.py` during `python scripts/build_corpus.py`. Full-corpus tensors live in `data/processed/corpus.npz`.
