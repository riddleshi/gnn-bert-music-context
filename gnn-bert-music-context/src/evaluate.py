"""End-to-end evaluation, ablations, plots, and demo-asset export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE

from src.bert_encoder import BertTagHead, TinyMusicBERT, Vocab
from src.cnn_baseline import MelCNN
from src.config import ROOT, load_config, resolve_device
from src.contrastive import DualEncoder, retrieval_metrics
from src.dataset import make_loaders
from src.fusion_model import GNNbertFusion
from src.gnn_model import GNNClassifier, GNNEncoder
from src.graph_builder import graph_coherence
from src.metrics import emotion_metrics, genre_accuracy, majority_baseline, tag_metrics
from src.taxonomy import GENRES, TAGS
from src.train import _make_gnn, _move, collect
from src.visualize import (
    save_ablation_bars,
    save_f1_curves,
    save_retrieval_panel,
    save_tsne,
    save_va_plane,
)


def _down2d(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = arr.shape
    ys = np.linspace(0, h, min(out_h, h), endpoint=False).astype(int)
    xs = np.linspace(0, w, min(out_w, w), endpoint=False).astype(int)
    return np.round(arr[ys][:, xs], 3).astype(np.float32)


def _load_state(model: torch.nn.Module, path: Path, device: str) -> torch.nn.Module:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def reconstruct_models(cfg, vocab: Vocab, in_dim: int, device: str) -> dict[str, Any]:
    ckpt = ROOT / cfg["paths"]["checkpoints"]
    hidden = int(cfg["model"]["hidden_dim"])
    models: dict[str, Any] = {}

    bert = TinyMusicBERT(
        vocab_size=len(vocab),
        hidden=int(cfg["model"]["bert_hidden"]),
        layers=int(cfg["model"]["bert_layers"]),
        heads=int(cfg["model"]["bert_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        max_len=int(cfg["max_text_len"]),
    )
    t1 = BertTagHead(bert, n_tags=len(TAGS), dropout=float(cfg["model"]["dropout"]))
    models["bert"] = _load_state(t1, ckpt / "task1_bert.pt", device)

    gnn_g = GNNClassifier(_make_gnn(cfg, in_dim), n_labels=len(GENRES), dropout=float(cfg["model"]["dropout"]))
    models["gnn_genre"] = _load_state(gnn_g, ckpt / "task2_gnn_genre.pt", device)

    gnn_t = GNNClassifier(_make_gnn(cfg, in_dim), n_labels=len(TAGS), dropout=float(cfg["model"]["dropout"]))
    models["gnn_tags"] = _load_state(gnn_t, ckpt / "task2_gnn_tags.pt", device)

    cnn_g = MelCNN(n_labels=len(GENRES), channels=list(cfg["model"]["cnn_channels"]))
    models["cnn_genre"] = _load_state(cnn_g, ckpt / "cnn_genre.pt", device)
    cnn_t = MelCNN(n_labels=len(TAGS), channels=list(cfg["model"]["cnn_channels"]))
    models["cnn_tags"] = _load_state(cnn_t, ckpt / "cnn_tags.pt", device)

    for mode in ("cross_attention", "concat"):
        gnn = _make_gnn(cfg, in_dim)
        bert_f = TinyMusicBERT(
            vocab_size=len(vocab),
            hidden=int(cfg["model"]["bert_hidden"]),
            layers=int(cfg["model"]["bert_layers"]),
            heads=int(cfg["model"]["bert_heads"]),
            dropout=float(cfg["model"]["dropout"]),
            max_len=int(cfg["max_text_len"]),
        )
        fusion = GNNbertFusion(gnn, bert_f, n_tags=len(TAGS), mode=mode, dropout=float(cfg["model"]["dropout"]))
        models[f"fusion_{mode}"] = _load_state(fusion, ckpt / f"task3_{mode}.pt", device)

    dual = DualEncoder(_make_gnn(cfg, in_dim), TinyMusicBERT(
        vocab_size=len(vocab),
        hidden=int(cfg["model"]["bert_hidden"]),
        layers=int(cfg["model"]["bert_layers"]),
        heads=int(cfg["model"]["bert_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        max_len=int(cfg["max_text_len"]),
    ), proj_dim=hidden)
    models["dual"] = _load_state(dual, ckpt / "task4_contrastive.pt", device)
    return models


def evaluate_all(cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    device = resolve_device(cfg)
    processed = ROOT / cfg["paths"]["processed"]
    vocab = Vocab.load(processed / "vocab.json")
    loaders = make_loaders(vocab, graph="segment", cfg=cfg)
    in_dim = next(iter(loaders["train"])).x.shape[-1]
    models = reconstruct_models(cfg, vocab, in_dim, device)

    test = loaders["test"]
    train_tags = np.concatenate([b.tags.numpy() for b in loaders["train"]], axis=0)
    n_test = sum(len(b.ids) for b in test)

    report: dict[str, Any] = {"n_test": n_test, "models": {}}

    # Majority / random baselines on tags.
    maj = majority_baseline(train_tags, n_test)
    maj_logits = np.log(np.clip(maj, 1e-4, 1 - 1e-4) / np.clip(1 - maj, 1e-4, 1))
    y_test = np.concatenate([b.tags.numpy() for b in test], axis=0)
    report["models"]["majority"] = tag_metrics(y_test, maj_logits)
    rng = np.random.default_rng(0)
    rand_logits = np.log(rng.random((n_test, len(TAGS))) + 1e-3)
    report["models"]["random"] = tag_metrics(y_test, rand_logits)

    # BERT-only tags
    blob = collect(lambda b: {"tag_logits": models["bert"](b.input_ids)}, test, device)
    report["models"]["bert_only"] = tag_metrics(blob["tags"], blob["tag_logits"])
    bert_examples = _tag_examples(loaders["_tracks"], blob)

    # GNN-only tags + genre
    blob_g = collect(lambda b: {"tag_logits": models["gnn_tags"](b.x, b.adj, b.mask)}, test, device)
    report["models"]["gnn_only"] = tag_metrics(blob_g["tags"], blob_g["tag_logits"])
    blob_gg = collect(lambda b: {"genre_logits": models["gnn_genre"](b.x, b.adj, b.mask)}, test, device)
    report["models"]["gnn_only"]["genre_acc"] = genre_accuracy(blob_gg["genre"], blob_gg["genre_logits"])

    blob_c = collect(lambda b: {"tag_logits": models["cnn_tags"](b.mel), "genre_logits": models["cnn_genre"](b.mel)}, test, device)
    # CNN tags and genre are separate nets — collect twice.
    blob_ct = collect(lambda b: {"tag_logits": models["cnn_tags"](b.mel)}, test, device)
    blob_cg = collect(lambda b: {"genre_logits": models["cnn_genre"](b.mel)}, test, device)
    report["models"]["cnn_mel"] = tag_metrics(blob_ct["tags"], blob_ct["tag_logits"])
    report["models"]["cnn_mel"]["genre_acc"] = genre_accuracy(blob_cg["genre"], blob_cg["genre_logits"])
    report["models"]["gnn_only"]["genre_acc"] = genre_accuracy(blob_gg["genre"], blob_gg["genre_logits"])

    fusion_blobs = {}
    for mode in ("cross_attention", "concat"):
        m = models[f"fusion_{mode}"]
        blob_f = collect(lambda b, mm=m: mm(b.x, b.adj, b.mask, b.input_ids), test, device)
        fusion_blobs[mode] = blob_f
        tags = tag_metrics(blob_f["tags"], blob_f["tag_logits"])
        emo = emotion_metrics(blob_f["v_true"], blob_f["valence"], blob_f["a_true"], blob_f["arousal"])
        report["models"][f"fusion_{mode}"] = {**tags, **emo}

    # Contrastive retrieval on the full test split.
    dual = models["dual"]
    zs_g, zs_t, ids = [], [], []
    with torch.no_grad():
        for batch in test:
            batch = _move(batch, device)
            zs_g.append(dual.encode_graph(batch.x, batch.adj, batch.mask).cpu())
            zs_t.append(dual.encode_text(batch.input_ids).cpu())
            ids.extend(batch.ids)
    g = torch.cat(zs_g)
    t = torch.cat(zs_t)
    retr = retrieval_metrics(g, t)
    report["models"]["contrastive"] = retr

    # Zero-shot tags from contrastive: nearest train caption prototypes — skip;
    # instead score cosine(graph, tag-name encoding) is not available. Report R@K only.

    history = json.loads((ROOT / cfg["paths"]["results"] / "history.json").read_text())
    plots = ROOT / cfg["paths"]["plots"]
    plots.mkdir(parents=True, exist_ok=True)
    save_f1_curves(history, plots / "f1_curves.png")
    save_ablation_bars(report["models"], plots / "ablation.png")

    z = fusion_blobs["cross_attention"]["z"]
    genres = fusion_blobs["cross_attention"]["genre"]
    va = np.stack(
        [fusion_blobs["cross_attention"]["v_true"], fusion_blobs["cross_attention"]["a_true"]],
        axis=1,
    )
    if z.shape[0] >= 8:
        z2 = TSNE(n_components=2, perplexity=min(12, max(5, z.shape[0] // 4)), random_state=0).fit_transform(z)
        save_tsne(z2, [GENRES[i] for i in genres], plots / "tsne_genre.png", "t-SNE of fusion vector z (colour = genre)")
        mood = np.where(va[:, 1] > 0.55, np.where(va[:, 0] > 0.5, "energetic-pos", "energetic-neg"), np.where(va[:, 0] > 0.5, "calm-pos", "calm-neg"))
        save_tsne(z2, mood.tolist(), plots / "tsne_mood.png", "t-SNE of fusion vector z (colour = valence/arousal quadrant)")
    save_va_plane(
        fusion_blobs["cross_attention"]["v_true"],
        fusion_blobs["cross_attention"]["a_true"],
        fusion_blobs["cross_attention"]["valence"],
        fusion_blobs["cross_attention"]["arousal"],
        [GENRES[i] for i in genres],
        plots / "valence_arousal.png",
    )

    by_id = {t["id"]: t for t in loaders["_tracks"]}
    retrieval_rows = _retrieval_examples(ids, g, t, by_id)
    save_retrieval_panel(retrieval_rows[:6], plots / "retrieval_examples.png")
    (ROOT / "results" / "retrieval_examples" / "examples.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "retrieval_examples" / "examples.json").write_text(json.dumps(retrieval_rows, indent=2))

    demo = build_demo_payload(
        cfg,
        loaders,
        models,
        fusion_blobs["cross_attention"],
        report,
        retrieval_rows,
        bert_examples,
        vocab,
        device,
    )
    demo_path = ROOT / cfg["paths"]["demo_json"]
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_text(json.dumps(demo))
    # Also keep a copy under results/
    (ROOT / "results" / "demo.json").write_text(json.dumps(demo, indent=2))

    import shutil

    pub = ROOT / "web" / "public" / "plots"
    pub.mkdir(parents=True, exist_ok=True)
    for p in plots.glob("*.png"):
        shutil.copy(p, pub / p.name)

    (ROOT / "results" / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def _tag_examples(tracks, blob, k: int = 5) -> list[dict[str, Any]]:
    by_id = {t["id"]: t for t in tracks}
    out = []
    probs = 1 / (1 + np.exp(-blob["tag_logits"]))
    for i, tid in enumerate(blob["ids"][:k]):
        t = by_id[tid]
        pred_idx = np.where(probs[i] >= 0.5)[0].tolist()
        out.append(
            {
                "id": tid,
                "caption": t["caption"],
                "true": t["tag_names"],
                "pred": [TAGS[j] for j in pred_idx],
                "scores": {TAGS[j]: float(probs[i, j]) for j in np.argsort(-probs[i])[:8]},
            }
        )
    return out


def _retrieval_examples(ids, g, t, by_id, n_queries: int = 10, k: int = 3) -> list[dict[str, Any]]:
    sim = F.normalize(g, dim=-1) @ F.normalize(t, dim=-1).T
    ranked = sim.argsort(dim=1, descending=True)
    rows = []
    # Query = caption i, retrieve audio (rows of sim.T would be caption→audio; sim is audio×text with aligned ids)
    # caption_to_audio uses sim.T[i] = similarities of caption i to all graphs.
    sim_c2a = sim.T
    ranked_c = sim_c2a.argsort(dim=1, descending=True)
    for i in range(min(n_queries, len(ids))):
        qid = ids[i]
        tops = []
        for j in ranked_c[i, :k].tolist():
            hid = ids[j]
            tops.append(
                {
                    "id": hid,
                    "title": by_id[hid]["title"],
                    "genre": by_id[hid]["genre"],
                    "caption": by_id[hid]["caption"],
                    "score": float(sim_c2a[i, j]),
                    "correct": hid == qid,
                }
            )
        rows.append(
            {
                "query_id": qid,
                "query_caption": by_id[qid]["caption"],
                "query_genre": by_id[qid]["genre"],
                "query_title": by_id[qid]["title"],
                "hits": tops,
            }
        )
    return rows


def build_demo_payload(cfg, loaders, models, fusion_blob, report, retrieval_rows, bert_examples, vocab, device) -> dict[str, Any]:
    tracks = loaders["_tracks"]
    by_id = {t["id"]: t for t in tracks}
    # Build a lookup from fusion test blob
    f_ids = fusion_blob["ids"]
    f_index = {tid: i for i, tid in enumerate(f_ids)}
    probs = 1 / (1 + np.exp(-fusion_blob["tag_logits"]))

    # Case studies: pick one jazz, one metal, one folk from test if possible.
    test_ids = loaders["_splits"]["test"]
    case_ids = []
    for want in ("jazz", "metal", "folk"):
        found = next((tid for tid in test_ids if by_id[tid]["genre"] == want), None)
        if found:
            case_ids.append(found)
    while len(case_ids) < 3 and test_ids:
        extra = test_ids[len(case_ids)]
        if extra not in case_ids:
            case_ids.append(extra)

    demo_tracks = []
    # Include a diverse subset (all test + a few train) for the explorer.
    show_ids = list(dict.fromkeys(test_ids + loaders["_splits"]["val"][:8]))
    fusion_model = models["fusion_cross_attention"]
    gnn_genre = models["gnn_genre"]

    for tid in show_ids:
        t = by_id[tid]
        item = {
            "id": tid,
            "title": t["title"],
            "genre": t["genre"],
            "caption": t["caption"],
            "tags": t["tag_names"],
            "valence": float(t["valence"]),
            "arousal": float(t["arousal"]),
            "tempo": float(t["tempo"]),
            "chord_seq": list(t["chord_seq"]),
            "segment_graph": t["seg_json"],
            "chord_graph": t["chord_json"],
            "mel": _down2d(np.asarray(t["mel_small"]), 32, 16).tolist(),
            "split": "test" if tid in loaders["_splits"]["test"] else "val" if tid in loaders["_splits"]["val"] else "train",
        }
        if tid in f_index:
            i = f_index[tid]
            pred_idx = np.where(probs[i] >= 0.5)[0].tolist()
            item["pred_tags"] = [TAGS[j] for j in pred_idx]
            item["pred_tag_scores"] = [
                {"tag": TAGS[j], "score": float(probs[i, j])}
                for j in np.argsort(-probs[i])[:10]
            ]
            item["pred_valence"] = float(fusion_blob["valence"][i])
            item["pred_arousal"] = float(fusion_blob["arousal"][i])
            item["z"] = [float(x) for x in fusion_blob["z"][i][:16]]
        else:
            # Live forward for val tracks.
            x = torch.from_numpy(np.asarray(t["seg_x"], dtype=np.float32)).unsqueeze(0).to(device)
            adj = torch.from_numpy(np.asarray(t["seg_adj"], dtype=np.float32)).unsqueeze(0).to(device)
            mask = torch.ones(1, x.shape[1], device=device)
            ids = torch.from_numpy(vocab.encode(t["caption"], int(cfg["max_text_len"]))).unsqueeze(0).to(device)
            with torch.no_grad():
                out = fusion_model(x, adj, mask, ids)
                g_logits = gnn_genre(x, adj, mask)
            p = torch.sigmoid(out["tag_logits"][0]).cpu().numpy()
            pred_idx = np.where(p >= 0.5)[0].tolist()
            item["pred_tags"] = [TAGS[j] for j in pred_idx]
            item["pred_tag_scores"] = [{"tag": TAGS[j], "score": float(p[j])} for j in np.argsort(-p)[:10]]
            item["pred_valence"] = float(out["valence"][0])
            item["pred_arousal"] = float(out["arousal"][0])
            item["pred_genre"] = GENRES[int(g_logits[0].argmax())]
            item["z"] = [float(x) for x in out["z"][0].cpu().numpy()[:16]]
        if "pred_genre" not in item:
            x = torch.from_numpy(np.asarray(t["seg_x"], dtype=np.float32)).unsqueeze(0).to(device)
            adj = torch.from_numpy(np.asarray(t["seg_adj"], dtype=np.float32)).unsqueeze(0).to(device)
            mask = torch.ones(1, x.shape[1], device=device)
            with torch.no_grad():
                g_logits = gnn_genre(x, adj, mask)
            item["pred_genre"] = GENRES[int(g_logits[0].argmax())]
        # Graph coherence on segment features.
        from src.graph_builder import MusicGraph

        mg = MusicGraph(
            x=np.asarray(t["seg_x"]),
            adj=np.asarray(t["seg_adj"]),
            edge_index=np.array([[e["source"] for e in t["seg_json"]["edges"]], [e["target"] for e in t["seg_json"]["edges"]]], dtype=np.int64)
            if t["seg_json"]["edges"]
            else np.zeros((2, 0), dtype=np.int64),
            edge_weight=np.array([e["weight"] for e in t["seg_json"]["edges"]], dtype=np.float32)
            if t["seg_json"]["edges"]
            else np.zeros((0,), dtype=np.float32),
            node_labels=[n["label"] for n in t["seg_json"]["nodes"]],
            kind="segment",
        )
        item["graph_coherence"] = graph_coherence(mg, tau=0.5)
        demo_tracks.append(item)

    history = json.loads((ROOT / cfg["paths"]["results"] / "history.json").read_text())
    return {
        "project": {
            "title": "GNN-Based BERT for Understanding Context from Music",
            "course": "CSE425 / EEE474 / CSE715 — Neural Networks",
            "tasks": [
                {"id": 1, "name": "BERT tag classifier", "marks": 18},
                {"id": 2, "name": "GNN on segment/chord graphs", "marks": 22},
                {"id": 3, "name": "GNN–BERT fusion", "marks": 22},
                {"id": 4, "name": "Contrastive MusicCaps alignment", "marks": 18},
            ],
        },
        "taxonomy": {"genres": GENRES, "tags": TAGS},
        "metrics": report,
        "history": history,
        "tracks": demo_tracks,
        "retrieval": retrieval_rows,
        "bert_examples": bert_examples,
        "case_ids": case_ids,
        "splits": {k: len(v) for k, v in loaders["_splits"].items()},
    }


if __name__ == "__main__":
    evaluate_all()
