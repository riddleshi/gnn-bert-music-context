"""FastAPI inference + demo payload server."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.audio_features import extract_bundle
from src.bert_encoder import Vocab
from src.config import ROOT, load_config, resolve_device
from src.evaluate import reconstruct_models
from src.graph_builder import build_graphs
from src.load_graph import graph_to_draw_json, load_pt_graph
from src.synthetic import sample_emotion, sample_tags, synthesize_clip
from src.taxonomy import GENRES, TAGS

cfg = load_config()
device = resolve_device(cfg)
DEMO_PATH = ROOT / cfg["paths"]["demo_json"]
VOCAB_PATH = ROOT / cfg["paths"]["processed"] / "vocab.json"

app = FastAPI(title="Cadence Lab — GNN–BERT Music Context", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODELS: dict[str, Any] | None = None
_VOCAB: Vocab | None = None
_IN_DIM = 0


def _models():
    global _MODELS, _VOCAB, _IN_DIM
    if _MODELS is None:
        if not VOCAB_PATH.exists():
            raise RuntimeError("Train the models first (python -m src.train --task all).")
        _VOCAB = Vocab.load(VOCAB_PATH)
        blob = np.load(ROOT / cfg["paths"]["processed"] / "corpus.npz", allow_pickle=True)
        _IN_DIM = int(blob["tracks"][0]["seg_x"].shape[1])
        _MODELS = reconstruct_models(cfg, _VOCAB, _IN_DIM, device)
    return _MODELS, _VOCAB


@app.get("/health")
def health():
    return {"ok": True, "device": device}


@app.get("/demo")
def demo():
    if not DEMO_PATH.exists():
        raise HTTPException(503, "demo.json missing — run src.evaluate after training.")
    return json.loads(DEMO_PATH.read_text())


@app.get("/graph/{name}")
def graph_from_pt(name: str):
    """Load a saved sample with torch.load and return a drawable graph."""
    import re

    if not re.fullmatch(r"track_\d{3}_(segment|chord)\.pt", name):
        raise HTTPException(400, "expected track_XXX_segment.pt or track_XXX_chord.pt")
    path = ROOT / "data" / "processed" / "graphs" / name
    if not path.exists():
        raise HTTPException(404, f"{name} not found")
    blob = load_pt_graph(path)
    drawn = graph_to_draw_json(blob)
    drawn["file"] = name
    drawn["keys"] = list(blob.keys())
    drawn["node_labels"] = list(blob["node_labels"])
    drawn["x_shape"] = list(blob["x"].shape)
    drawn["code"] = (
        "import torch\n\n"
        f'graph = torch.load("data/processed/graphs/{name}", weights_only=False)\n\n'
        "print(graph.keys())\n"
        'print(graph["node_labels"])\n'
    )
    return drawn


class PredictIn(BaseModel):
    caption: str = Field(..., min_length=8, max_length=400)
    genre: str | None = None
    seed: int = 0


@app.post("/predict")
def predict(body: PredictIn):
    models, vocab = _models()
    rng = np.random.default_rng(body.seed)
    genre = body.genre if body.genre in GENRES else GENRES[int(rng.integers(0, len(GENRES)))]
    sr = int(cfg["sample_rate"])
    y, chord_seq, tempo = synthesize_clip(rng, genre, float(cfg["duration_seconds"]), sr)
    valence, arousal = sample_emotion(rng, genre)
    tags = sample_tags(rng, genre)
    caption = body.caption.strip()
    bundle = extract_bundle(y, sr)
    graphs = build_graphs(bundle)
    x = torch.from_numpy(graphs["segment"].x).unsqueeze(0).to(device)
    adj = torch.from_numpy(graphs["segment"].adj).unsqueeze(0).to(device)
    mask = torch.ones(1, x.shape[1], device=device)
    ids = torch.from_numpy(vocab.encode(caption, int(cfg["max_text_len"]))).unsqueeze(0).to(device)
    fusion = models["fusion_cross_attention"]
    gnn_genre = models["gnn_genre"]
    bert = models["bert"]
    with torch.no_grad():
        out = fusion(x, adj, mask, ids)
        genre_logits = gnn_genre(x, adj, mask)
        tag_logits = bert(ids)
    p = torch.sigmoid(out["tag_logits"][0]).cpu().numpy()
    p_bert = torch.sigmoid(tag_logits[0]).cpu().numpy()
    pred_idx = np.where(p >= 0.5)[0].tolist()
    return {
        "genre_true_proxy": genre,
        "pred_genre": GENRES[int(genre_logits[0].argmax())],
        "caption": caption,
        "chord_seq": chord_seq,
        "tempo": tempo,
        "true_tags_proxy": [TAGS[i] for i, v in enumerate(tags) if v > 0],
        "pred_tags": [TAGS[i] for i in pred_idx],
        "pred_tag_scores": [{"tag": TAGS[i], "score": float(p[i])} for i in np.argsort(-p)[:10]],
        "bert_only_tags": [TAGS[i] for i in np.where(p_bert >= 0.5)[0].tolist()],
        "pred_valence": float(out["valence"][0]),
        "pred_arousal": float(out["arousal"][0]),
        "proxy_valence": valence,
        "proxy_arousal": arousal,
        "segment_graph": graphs["segment"].to_json(),
        "chord_graph": graphs["chord"].to_json(),
        "mel": _down(bundle.log_mel, 64, 32).tolist(),
        "attn": out["attn"][0].cpu().numpy().tolist() if out.get("attn") is not None else None,
    }


def _down(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    ys = np.linspace(0, arr.shape[0], h, endpoint=False).astype(int)
    xs = np.linspace(0, arr.shape[1], w, endpoint=False).astype(int)
    return arr[ys][:, xs]


def main() -> None:
    uvicorn.run("api.server:app", host="0.0.0.0", port=43181, reload=False)


if __name__ == "__main__":
    main()
