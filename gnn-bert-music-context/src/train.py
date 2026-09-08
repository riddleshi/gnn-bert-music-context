"""Train Tasks 1–4 plus CNN / majority baselines.

Usage:
    python -m src.train --task all
    python -m src.train --task 3 --fusion concat
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.bert_encoder import BertTagHead, TinyMusicBERT, Vocab, build_vocab
from src.cnn_baseline import MelCNN
from src.config import ROOT, load_config, resolve_device
from src.contrastive import DualEncoder, retrieval_metrics
from src.dataset import Batch, make_loaders
from src.fusion_model import GNNbertFusion
from src.gnn_model import GNNClassifier, GNNEncoder
from src.metrics import emotion_metrics, genre_accuracy, majority_baseline, tag_metrics
from src.taxonomy import GENRES, TAGS


def _move(batch: Batch, device: str) -> Batch:
    return Batch(
        input_ids=batch.input_ids.to(device),
        x=batch.x.to(device),
        adj=batch.adj.to(device),
        mask=batch.mask.to(device),
        tags=batch.tags.to(device),
        genre=batch.genre.to(device),
        valence=batch.valence.to(device),
        arousal=batch.arousal.to(device),
        mel=batch.mel.to(device),
        ids=batch.ids,
        captions=batch.captions,
    )


@torch.no_grad()
def collect(model_fn: Callable[[Batch], dict[str, torch.Tensor]], loader: DataLoader, device: str) -> dict[str, np.ndarray]:
    acc: dict[str, list] = {
        "tag_logits": [],
        "genre_logits": [],
        "valence": [],
        "arousal": [],
        "tags": [],
        "genre": [],
        "v_true": [],
        "a_true": [],
        "z": [],
        "ids": [],
    }
    model_fn_mod = model_fn
    for batch in loader:
        batch = _move(batch, device)
        out = model_fn_mod(batch)
        if "tag_logits" in out:
            acc["tag_logits"].append(out["tag_logits"].cpu().numpy())
        if "genre_logits" in out:
            acc["genre_logits"].append(out["genre_logits"].cpu().numpy())
        if "valence" in out:
            acc["valence"].append(out["valence"].cpu().numpy())
            acc["arousal"].append(out["arousal"].cpu().numpy())
        if "z" in out:
            acc["z"].append(out["z"].cpu().numpy())
        acc["tags"].append(batch.tags.cpu().numpy())
        acc["genre"].append(batch.genre.cpu().numpy())
        acc["v_true"].append(batch.valence.cpu().numpy())
        acc["a_true"].append(batch.arousal.cpu().numpy())
        acc["ids"].extend(batch.ids)
    stacked = {}
    for k, v in acc.items():
        if k == "ids":
            stacked[k] = v
        elif v and isinstance(v[0], np.ndarray):
            stacked[k] = np.concatenate(v, axis=0)
    return stacked


def eval_tags(model_fn, loader, device) -> dict[str, float]:
    blob = collect(model_fn, loader, device)
    return tag_metrics(blob["tags"], blob["tag_logits"])


def _optimizer(params, cfg) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))


def train_task1(cfg, loaders, device, vocab: Vocab) -> tuple[BertTagHead, list[dict]]:
    enc = TinyMusicBERT(
        vocab_size=len(vocab),
        hidden=int(cfg["model"]["bert_hidden"]),
        layers=int(cfg["model"]["bert_layers"]),
        heads=int(cfg["model"]["bert_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        max_len=int(cfg["max_text_len"]),
    )
    model = BertTagHead(enc, n_tags=len(TAGS), dropout=float(cfg["model"]["dropout"])).to(device)
    opt = _optimizer(model.parameters(), cfg)
    history = []
    best, best_state = -1.0, None
    epochs = int(cfg["train"]["epochs_task1"])
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(loaders["train"], desc=f"task1 epoch {epoch}", leave=False):
            batch = _move(batch, device)
            logits = model(batch.input_ids)
            loss = F.binary_cross_entropy_with_logits(logits, batch.tags)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        val = eval_tags(lambda b: {"tag_logits": model(b.input_ids)}, loaders["val"], device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(f"  Task1 epoch {epoch}: loss={row['loss']:.4f} val macro-F1={val['macro_f1']:.3f}")
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
    if best_state:
        model.load_state_dict(best_state)
    return model, history


def _make_gnn(cfg, in_dim: int, kind: str | None = None) -> GNNEncoder:
    return GNNEncoder(
        in_dim=in_dim,
        hidden=int(cfg["model"]["hidden_dim"]),
        layers=int(cfg["model"]["gnn_layers"]),
        kind=kind or cfg["model"]["gnn_type"],
        heads=int(cfg["model"]["gnn_heads"]),
        dropout=float(cfg["model"]["dropout"]),
    )


def train_task2(cfg, loaders, device, n_labels: int, target: str = "genre") -> tuple[GNNClassifier, list[dict]]:
    in_dim = next(iter(loaders["train"])).x.shape[-1]
    enc = _make_gnn(cfg, in_dim)
    model = GNNClassifier(enc, n_labels=n_labels, dropout=float(cfg["model"]["dropout"])).to(device)
    opt = _optimizer(model.parameters(), cfg)
    history = []
    best, best_state = -1.0, None
    epochs = int(cfg["train"]["epochs_task2"])
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(loaders["train"], desc=f"task2 {target} epoch {epoch}", leave=False):
            batch = _move(batch, device)
            logits = model(batch.x, batch.adj, batch.mask)
            if target == "genre":
                loss = F.cross_entropy(logits, batch.genre)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, batch.tags)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        blob = collect(lambda b: {"genre_logits": model(b.x, b.adj, b.mask), "tag_logits": model(b.x, b.adj, b.mask)}, loaders["val"], device)
        if target == "genre":
            score = genre_accuracy(blob["genre"], blob["genre_logits"])
            extra = {"val_acc": score}
        else:
            extra = tag_metrics(blob["tags"], blob["tag_logits"])
            extra = {f"val_{k}": v for k, v in extra.items()}
            score = extra["val_macro_f1"]
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **extra}
        history.append(row)
        print(f"  Task2 epoch {epoch}: loss={row['loss']:.4f} {extra}")
        if score > best:
            best = score
            best_state = copy.deepcopy(model.state_dict())
    if best_state:
        model.load_state_dict(best_state)
    return model, history


def train_cnn(cfg, loaders, device, n_labels: int, target: str = "genre") -> tuple[MelCNN, list[dict]]:
    model = MelCNN(n_labels=n_labels, in_h=64, in_w=32, channels=list(cfg["model"]["cnn_channels"])).to(device)
    opt = _optimizer(model.parameters(), cfg)
    history = []
    best, best_state = -1.0, None
    for epoch in range(1, int(cfg["train"]["epochs_task2"]) + 1):
        model.train()
        losses = []
        for batch in tqdm(loaders["train"], desc=f"cnn epoch {epoch}", leave=False):
            batch = _move(batch, device)
            logits = model(batch.mel)
            loss = F.cross_entropy(logits, batch.genre) if target == "genre" else F.binary_cross_entropy_with_logits(logits, batch.tags)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        blob = collect(lambda b: {"genre_logits": model(b.mel), "tag_logits": model(b.mel)}, loaders["val"], device)
        if target == "genre":
            score = genre_accuracy(blob["genre"], blob["genre_logits"])
            extra = {"val_acc": score}
        else:
            extra = {f"val_{k}": v for k, v in tag_metrics(blob["tags"], blob["tag_logits"]).items()}
            score = extra["val_macro_f1"]
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **extra}
        history.append(row)
        print(f"  CNN epoch {epoch}: {row}")
        if score > best:
            best = score
            best_state = copy.deepcopy(model.state_dict())
    if best_state:
        model.load_state_dict(best_state)
    return model, history


def train_fusion(cfg, loaders, device, vocab: Vocab, mode: str) -> tuple[GNNbertFusion, list[dict]]:
    in_dim = next(iter(loaders["train"])).x.shape[-1]
    gnn = _make_gnn(cfg, in_dim)
    bert = TinyMusicBERT(
        vocab_size=len(vocab),
        hidden=int(cfg["model"]["bert_hidden"]),
        layers=int(cfg["model"]["bert_layers"]),
        heads=int(cfg["model"]["bert_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        max_len=int(cfg["max_text_len"]),
    )
    model = GNNbertFusion(gnn, bert, n_tags=len(TAGS), mode=mode, dropout=float(cfg["model"]["dropout"])).to(device)
    t1_path = ROOT / cfg["paths"]["checkpoints"] / "task1_bert.pt"
    if t1_path.exists():
        state = torch.load(t1_path, map_location=device, weights_only=True)
        enc_state = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = model.bert.load_state_dict(enc_state, strict=False)
        print(f"  warm-start BERT from Task 1 ({mode}); missing={len(missing)} unexpected={len(unexpected)}")
    opt = _optimizer(model.parameters(), cfg)
    alpha = float(cfg["train"]["alpha_emotion"])
    beta = float(cfg["train"]["beta_emotion"])
    history = []
    best, best_state = -1e9, None
    for epoch in range(1, int(cfg["train"]["epochs_task3"]) + 1):
        model.train()
        losses = []
        for batch in tqdm(loaders["train"], desc=f"fusion-{mode} epoch {epoch}", leave=False):
            batch = _move(batch, device)
            out = model(batch.x, batch.adj, batch.mask, batch.input_ids)
            l_tags = F.binary_cross_entropy_with_logits(out["tag_logits"], batch.tags)
            l_v = F.mse_loss(out["valence"], batch.valence)
            l_a = F.mse_loss(out["arousal"], batch.arousal)
            loss = l_tags + alpha * l_v + beta * l_a
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        blob = collect(
            lambda b: model(b.x, b.adj, b.mask, b.input_ids),
            loaders["val"],
            device,
        )
        tags = tag_metrics(blob["tags"], blob["tag_logits"])
        emo = emotion_metrics(blob["v_true"], blob["valence"], blob["a_true"], blob["arousal"])
        score = tags["macro_f1"] - 0.15 * emo["mae_emotion"]
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in {**tags, **emo}.items()}}
        history.append(row)
        print(f"  Fusion[{mode}] epoch {epoch}: F1={tags['macro_f1']:.3f} MAE={emo['mae_emotion']:.3f}")
        if score > best:
            best = score
            best_state = copy.deepcopy(model.state_dict())
    if best_state:
        model.load_state_dict(best_state)
    return model, history


def train_contrastive(cfg, loaders, device, vocab: Vocab) -> tuple[DualEncoder, list[dict]]:
    in_dim = next(iter(loaders["train"])).x.shape[-1]
    gnn = _make_gnn(cfg, in_dim)
    bert = TinyMusicBERT(
        vocab_size=len(vocab),
        hidden=int(cfg["model"]["bert_hidden"]),
        layers=int(cfg["model"]["bert_layers"]),
        heads=int(cfg["model"]["bert_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        max_len=int(cfg["max_text_len"]),
    )
    model = DualEncoder(gnn, bert, proj_dim=int(cfg["model"]["hidden_dim"])).to(device)
    opt = _optimizer(model.parameters(), cfg)
    tau = float(cfg["train"]["temperature"])
    history = []
    best, best_state = -1.0, None
    for epoch in range(1, int(cfg["train"]["epochs_task4"]) + 1):
        model.train()
        losses = []
        for batch in tqdm(loaders["train"], desc=f"contrastive epoch {epoch}", leave=False):
            batch = _move(batch, device)
            out = model(batch.x, batch.adj, batch.mask, batch.input_ids, temperature=tau)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            losses.append(float(out["loss"].detach()))
        model.eval()
        zs_g, zs_t = [], []
        with torch.no_grad():
            for batch in loaders["val"]:
                batch = _move(batch, device)
                zs_g.append(model.encode_graph(batch.x, batch.adj, batch.mask))
                zs_t.append(model.encode_text(batch.input_ids))
        g = torch.cat(zs_g)
        t = torch.cat(zs_t)
        retr = retrieval_metrics(g, t)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in retr.items()}}
        history.append(row)
        print(f"  Task4 epoch {epoch}: loss={row['loss']:.4f} R@5={retr['caption_to_audio_R@5']:.3f}")
        if retr["caption_to_audio_R@5"] > best:
            best = retr["caption_to_audio_R@5"]
            best_state = copy.deepcopy(model.state_dict())
    if best_state:
        model.load_state_dict(best_state)
    return model, history


def save_ckpt(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=["1", "2", "3", "4", "all", "cnn"])
    parser.add_argument("--fusion", default="cross_attention", choices=["cross_attention", "concat"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    device = resolve_device(cfg)
    print(f"device={device}")

    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    processed = ROOT / cfg["paths"]["processed"]

    tracks = __import__("src.synthetic", fromlist=["load_cached_corpus"]).load_cached_corpus(cfg)
    vocab_path = processed / "vocab.json"
    vocab = build_vocab([t["caption"] for t in tracks], max_size=int(cfg["model"]["vocab_size"]))
    vocab.dump(vocab_path)
    loaders = make_loaders(vocab, graph="segment", cfg=cfg)

    histories: dict[str, Any] = {}
    if args.task in {"1", "all"}:
        m1, h1 = train_task1(cfg, loaders, device, vocab)
        save_ckpt(m1, ckpt_dir / "task1_bert.pt")
        histories["task1"] = h1
    if args.task in {"2", "all"}:
        m2, h2 = train_task2(cfg, loaders, device, n_labels=len(GENRES), target="genre")
        save_ckpt(m2, ckpt_dir / "task2_gnn_genre.pt")
        histories["task2_genre"] = h2
        m2t, h2t = train_task2(cfg, loaders, device, n_labels=len(TAGS), target="tags")
        save_ckpt(m2t, ckpt_dir / "task2_gnn_tags.pt")
        histories["task2_tags"] = h2t
    if args.task in {"cnn", "all"}:
        cnn, hc = train_cnn(cfg, loaders, device, n_labels=len(GENRES), target="genre")
        save_ckpt(cnn, ckpt_dir / "cnn_genre.pt")
        histories["cnn_genre"] = hc
        cnn_t, hct = train_cnn(cfg, loaders, device, n_labels=len(TAGS), target="tags")
        save_ckpt(cnn_t, ckpt_dir / "cnn_tags.pt")
        histories["cnn_tags"] = hct
    if args.task in {"3", "all"}:
        for mode in ("cross_attention", "concat"):
            mf, hf = train_fusion(cfg, loaders, device, vocab, mode)
            save_ckpt(mf, ckpt_dir / f"task3_{mode}.pt")
            histories[f"task3_{mode}"] = hf
    if args.task in {"4", "all"}:
        m4, h4 = train_contrastive(cfg, loaders, device, vocab)
        save_ckpt(m4, ckpt_dir / "task4_contrastive.pt")
        histories["task4"] = h4

    hist_path = ROOT / cfg["paths"]["results"] / "history.json"
    prev: dict[str, Any] = {}
    if hist_path.exists():
        try:
            prev = json.loads(hist_path.read_text())
        except json.JSONDecodeError:
            prev = {}
    prev.update(histories)
    hist_path.write_text(json.dumps(prev, indent=2))
    print("training complete → results/history.json")


if __name__ == "__main__":
    main()
