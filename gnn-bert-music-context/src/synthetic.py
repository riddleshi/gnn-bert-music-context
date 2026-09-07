"""Synthetic paired audio / text / graph corpus.

Real FMA / MagnaTagATune / MusicCaps / DEAM downloads are large and gated.
This generator produces genre-conditioned 8 s clips, MusicCaps-style captions,
multi-label tags, and DEAM-style valence/arousal so every training script runs
offline. Swap in `src/real_data.py` loaders when the official dumps are present.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.audio_features import extract_bundle, node_feature_dim
from src.config import ROOT, load_config
from src.graph_builder import build_graphs
from src.taxonomy import (
    GENRE_PRIOR,
    GENRE_TO_IDX,
    GENRES,
    MOOD_WORDS,
    QUALITY_WORDS,
    TAG_TO_IDX,
    TAGS,
)

PC_FREQ = 440.0 * (2 ** ((np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) - 9) / 12.0))
# Map chord names like "Amin" / "Gmaj" onto pitch-class roots.
ROOT_PC = {n: i for i, n in enumerate("C C# D D# E F F# G G# A A# B".split())}
ROOT_PC.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})


def _chord_freqs(name: str) -> list[float]:
    quality = "min" if name.endswith("min") else "maj"
    root = name[: -3] if quality == "min" else name[: -3]
    if name.endswith("maj"):
        root = name[:-3]
        intervals = (0, 4, 7)
    else:
        root = name[:-3]
        intervals = (0, 3, 7)
    pc = ROOT_PC.get(root, 0)
    freqs = []
    for iv in intervals:
        freqs.append(float(PC_FREQ[(pc + iv) % 12]))
        freqs.append(float(PC_FREQ[(pc + iv) % 12] * 2))  # one octave up
    return freqs


def _envelope(n: int, attack: float = 0.02, sr: int = 22050) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    if n < 8:
        return env
    a = max(1, min(n // 4, int(attack * sr)))
    env[:a] = np.linspace(0, 1, a, dtype=np.float32)
    rel = max(1, min(n // 5, n - a))
    env[-rel:] *= np.linspace(1, 0.05, rel, dtype=np.float32)
    return env


def synthesize_clip(
    rng: np.random.Generator,
    genre: str,
    duration: float,
    sr: int,
) -> tuple[np.ndarray, list[str], float]:
    prior = GENRE_PRIOR[genre]
    tempo = float(rng.uniform(*prior["tempo"]))
    chords = list(prior["chords"])
    rng.shuffle(chords)
    # Loop 4–6 chords.
    seq = (chords + chords)[: rng.integers(4, 7)]
    n = int(duration * sr)
    t = np.arange(n) / sr
    y = np.zeros(n, dtype=np.float32)
    beat = 60.0 / tempo
    # Hold each chord for 2 beats (jazz/classical/folk) or 1–2 beats (faster styles).
    hold = 2 * beat if genre in {"jazz", "classical", "folk", "hip-hop"} else 1.5 * beat
    pos = 0.0
    i = 0
    brightness = {
        "jazz": 0.35,
        "rock": 0.55,
        "classical": 0.25,
        "electronic": 0.85,
        "hip-hop": 0.4,
        "pop": 0.7,
        "metal": 0.95,
        "folk": 0.3,
    }[genre]
    while pos < duration:
        chord = seq[i % len(seq)]
        length = min(hold, duration - pos)
        n_seg = int(length * sr)
        start = int(pos * sr)
        end = min(n, start + n_seg)
        n_seg = end - start
        if n_seg <= 4:
            break
        tt = t[start:end] - t[start]
        env = _envelope(n_seg, attack=0.01 if genre in {"metal", "rock"} else 0.03, sr=sr)
        partials = _chord_freqs(chord)
        tone = np.zeros(n_seg, dtype=np.float32)
        for p_i, f in enumerate(partials):
            amp = 0.18 / (1 + 0.35 * p_i)
            detune = 1.0 + float(rng.uniform(-0.002, 0.002))
            wave = np.sin(2 * np.pi * f * detune * tt)
            if brightness > 0.6:
                wave += 0.25 * np.sign(np.sin(2 * np.pi * f * tt))  # grit
            tone += (amp * wave).astype(np.float32)
        # Percussive pulses on beats.
        pulse = np.zeros(n_seg, dtype=np.float32)
        beat_hop = beat
        k = 0.0
        while k < length:
            idx = int(k * sr)
            if 0 <= idx < n_seg:
                width = int(0.03 * sr)
                sl = slice(idx, min(n_seg, idx + width))
                noise = rng.standard_normal(sl.stop - sl.start).astype(np.float32)
                kick = np.sin(2 * np.pi * 70 * np.arange(sl.stop - sl.start) / sr).astype(np.float32)
                pulse[sl] += 0.35 * kick + (0.12 + 0.2 * brightness) * noise * np.linspace(
                    1, 0, sl.stop - sl.start
                )
            k += beat_hop / (2 if genre in {"electronic", "metal", "rock"} else 1)
        y[start:end] += (tone * env + pulse * env) * 0.7
        pos += length
        i += 1

    # Genre-colored noise floor / sub bass.
    y += 0.02 * rng.standard_normal(n).astype(np.float32)
    if genre in {"hip-hop", "electronic"}:
        y += 0.12 * np.sin(2 * np.pi * 45 * t).astype(np.float32)
    peak = np.max(np.abs(y)) + 1e-6
    y = 0.9 * y / peak
    return y.astype(np.float32), seq, tempo


def sample_tags(rng: np.random.Generator, genre: str) -> np.ndarray:
    prior = GENRE_PRIOR[genre]["tags"]
    y = np.zeros(len(TAGS), dtype=np.float32)
    for tag, p in prior.items():
        if tag in TAG_TO_IDX and rng.random() < p:
            y[TAG_TO_IDX[tag]] = 1.0
    # A few global negatives stay 0; sprinkle rare extra tags.
    if rng.random() < 0.08:
        y[rng.integers(0, len(TAGS))] = 1.0
    if y.sum() < 3:
        # Guarantee a minimum of 3 positives so BCE is not degenerate.
        extras = [t for t in prior if t in TAG_TO_IDX]
        rng.shuffle(extras)
        for t in extras[:3]:
            y[TAG_TO_IDX[t]] = 1.0
    return y


def sample_caption(rng: np.random.Generator, genre: str, valence: float, arousal: float) -> str:
    templates = GENRE_PRIOR[genre]["captions"]
    tmpl = templates[int(rng.integers(0, len(templates)))]
    mood_pool = []
    mood_pool += MOOD_WORDS["high_valence" if valence >= 0.5 else "low_valence"]
    mood_pool += MOOD_WORDS["high_arousal" if arousal >= 0.5 else "low_arousal"]
    mood = mood_pool[int(rng.integers(0, len(mood_pool)))]
    quality = QUALITY_WORDS[int(rng.integers(0, len(QUALITY_WORDS)))]
    return tmpl.format(mood=mood, quality=quality)


def sample_emotion(rng: np.random.Generator, genre: str) -> tuple[float, float]:
    v0, v1 = GENRE_PRIOR[genre]["valence"]
    a0, a1 = GENRE_PRIOR[genre]["arousal"]
    v = float(np.clip(rng.uniform(v0, v1), 0, 1))
    a = float(np.clip(rng.uniform(a0, a1), 0, 1))
    return v, a


def build_corpus(cfg: dict[str, Any] | None = None, progress: bool = True) -> dict[str, Any]:
    cfg = cfg or load_config()
    rng = np.random.default_rng(int(cfg["seed"]))
    n_tracks = int(cfg["data"]["n_tracks"])
    n_artists = int(cfg["data"]["n_artists"])
    sr = int(cfg["sample_rate"])
    duration = float(cfg["duration_seconds"])

    tracks = []
    iterator = range(n_tracks)
    if progress:
        iterator = tqdm(iterator, desc="synthesize + featurize")
    for i in iterator:
        genre = GENRES[i % len(GENRES)]
        artist = i % n_artists
        y_audio, chord_seq, tempo = synthesize_clip(rng, genre, duration, sr)
        valence, arousal = sample_emotion(rng, genre)
        tags = sample_tags(rng, genre)
        caption = sample_caption(rng, genre, valence, arousal)
        bundle = extract_bundle(y_audio, sr)
        graphs = build_graphs(bundle)
        # Downsampled mel for the demo heatmap (64 x 32).
        mel = bundle.log_mel
        mel_small = _downsample2d(mel, 64, 32)
        chroma_small = _downsample2d(bundle.chroma, 12, 16)
        tracks.append(
            {
                "id": f"track_{i:03d}",
                "title": _title(rng, genre, i),
                "genre": genre,
                "genre_idx": GENRE_TO_IDX[genre],
                "artist_id": int(artist),
                "caption": caption,
                "tags": tags,
                "tag_names": [TAGS[j] for j, v in enumerate(tags) if v > 0],
                "valence": valence,
                "arousal": arousal,
                "tempo": tempo,
                "chord_seq": chord_seq,
                "segment_graph": graphs["segment"],
                "chord_graph": graphs["chord"],
                "mel_small": mel_small,
                "chroma_small": chroma_small,
                "n_mels_frames": int(bundle.log_mel.shape[1]),
            }
        )

    splits = artist_splits(tracks, cfg)
    return {"tracks": tracks, "splits": splits, "feat_dim": node_feature_dim()}


def artist_splits(tracks: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Artist-aware split: no artist appears in more than one partition."""
    rng = np.random.default_rng(int(cfg["seed"]) + 7)
    artists = sorted({t["artist_id"] for t in tracks})
    rng.shuffle(artists)
    n = len(artists)
    n_train = int(round(n * float(cfg["data"]["train_frac"])))
    n_val = int(round(n * float(cfg["data"]["val_frac"])))
    train_a = set(artists[:n_train])
    val_a = set(artists[n_train : n_train + n_val])
    test_a = set(artists[n_train + n_val :])
    splits = {"train": [], "val": [], "test": []}
    for t in tracks:
        if t["artist_id"] in train_a:
            splits["train"].append(t["id"])
        elif t["artist_id"] in val_a:
            splits["val"].append(t["id"])
        else:
            splits["test"].append(t["id"])
    return splits


def _downsample2d(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = arr.shape
    ys = np.linspace(0, h, out_h, endpoint=False).astype(int)
    xs = np.linspace(0, w, out_w, endpoint=False).astype(int)
    return arr[ys][:, xs].astype(np.float32)


def _title(rng: np.random.Generator, genre: str, i: int) -> str:
    adjectives = [
        "Midnight",
        "Copper",
        "Silent",
        "Neon",
        "Amber",
        "Hollow",
        "Velvet",
        "Iron",
        "Cedar",
        "Glass",
        "North",
        "Ember",
    ]
    nouns = [
        "Changes",
        "Stations",
        "Harbor",
        "Circuit",
        "Room",
        "Garden",
        "Wire",
        "Psalm",
        "Engine",
        "Window",
        "Current",
        "Letter",
    ]
    return f"{adjectives[i % len(adjectives)]} {nouns[int(rng.integers(0, len(nouns)))]}"


def persist_corpus(corpus: dict[str, Any], cfg: dict[str, Any] | None = None) -> None:
    """Write splits, 20+ example graphs, and a compact training cache."""
    cfg = cfg or load_config()
    processed = ROOT / cfg["paths"]["processed"]
    graphs_dir = ROOT / cfg["paths"]["graphs"]
    splits_dir = ROOT / cfg["paths"]["splits"]
    processed.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    cache = []
    for t in corpus["tracks"]:
        cache.append(
            {
                "id": t["id"],
                "genre_idx": t["genre_idx"],
                "genre": t["genre"],
                "artist_id": t["artist_id"],
                "caption": t["caption"],
                "tags": t["tags"],
                "tag_names": t["tag_names"],
                "valence": t["valence"],
                "arousal": t["arousal"],
                "title": t["title"],
                "tempo": t["tempo"],
                "chord_seq": t["chord_seq"],
                "mel_small": t["mel_small"],
                "chroma_small": t["chroma_small"],
                "seg_x": t["segment_graph"].x,
                "seg_adj": t["segment_graph"].adj,
                "chord_x": t["chord_graph"].x,
                "chord_adj": t["chord_graph"].adj,
                "seg_json": t["segment_graph"].to_json(),
                "chord_json": t["chord_graph"].to_json(),
            }
        )
    np.savez_compressed(processed / "corpus.npz", tracks=np.array(cache, dtype=object), feat_dim=corpus["feat_dim"])

    with (splits_dir / "splits.json").open("w") as f:
        json.dump(corpus["splits"], f, indent=2)

    # At least 20 example graphs (segment + chord for 12 tracks = 24 files).
    for t in corpus["tracks"][:12]:
        seg = t["segment_graph"]
        ch = t["chord_graph"]
        (graphs_dir / f"{t['id']}_segment.json").write_text(json.dumps(seg.to_json(), indent=2))
        (graphs_dir / f"{t['id']}_chord.json").write_text(json.dumps(ch.to_json(), indent=2))
        import torch

        torch.save(seg.to_torch(), graphs_dir / f"{t['id']}_segment.pt")
        torch.save(ch.to_torch(), graphs_dir / f"{t['id']}_chord.pt")

    manifest = {
        "n_tracks": len(corpus["tracks"]),
        "feat_dim": corpus["feat_dim"],
        "n_tags": len(TAGS),
        "n_genres": len(GENRES),
        "example_graphs": sorted(p.name for p in graphs_dir.glob("*")),
        "note": "Synthetic stand-in for FMA-small + MusicCaps + DEAM. See src/real_data.py.",
    }
    (processed / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_cached_corpus(cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = cfg or load_config()
    path = ROOT / cfg["paths"]["processed"] / "corpus.npz"
    blob = np.load(path, allow_pickle=True)
    tracks = list(blob["tracks"])
    return tracks
