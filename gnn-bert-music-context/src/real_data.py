"""Load official dumps (FMA, GTZAN, MusicCaps, DEAM, MagnaTagATune).

Drop files under ``data/raw/`` using the layout in ``data/raw/README.md``.
``scripts/build_corpus.py`` prefers this path whenever audio is present and
writes the same ``corpus.npz`` schema as the synthetic generator, so
``src.train`` does not change.

This module does **not** download anything. It only reads what you place in
``data/raw/``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

from src.audio_features import extract_bundle, load_mono, node_feature_dim
from src.config import ROOT, load_config
from src.graph_builder import build_graphs
from src.taxonomy import GENRE_PRIOR, GENRE_TO_IDX, GENRES, TAG_TO_IDX, TAGS

FMA_URL = "https://github.com/mdeff/fma"
MTT_URL = "https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset"
GTZAN_URL = "https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification"
DEAM_URL = "https://cvml.unige.ch/databases/DEAM/"
MUSICCAPS_URL = "https://www.kaggle.com/datasets/googleai/musiccaps"

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}

GENRE_ALIASES = {
    "hiphop": "hip-hop",
    "hip hop": "hip-hop",
    "hip-hop": "hip-hop",
    "rap": "hip-hop",
    "electro": "electronic",
    "electronic": "electronic",
    "edm": "electronic",
    "techno": "electronic",
    "house": "electronic",
    "disco": "electronic",
    "dance": "electronic",
    "ambient": "electronic",
    "experimental": "electronic",
    "instrumental": "classical",
    "classical": "classical",
    "opera": "classical",
    "jazz": "jazz",
    "blues": "folk",
    "country": "folk",
    "folk": "folk",
    "singer-songwriter": "folk",
    "international": "folk",
    "old-time / historic": "folk",
    "easy listening": "pop",
    "pop": "pop",
    "soul-rnb": "pop",
    "rnb": "pop",
    "soul": "pop",
    "rock": "rock",
    "punk": "rock",
    "indie": "rock",
    "metal": "metal",
    "spoken": "hip-hop",
}

TAG_SYNONYMS = {
    "guitar": ("guitar", "guitars", "acoustic guitar", "electric guitar"),
    "piano": ("piano", "keys", "keyboard"),
    "drums": ("drum", "drums", "beat", "percussion"),
    "synth": ("synth", "synthesizer", "pad", "analog"),
    "vocals": ("vocal", "vocals", "voice", "singing", "singer"),
    "strings": ("string", "strings", "violin", "cello", "orchestra"),
    "bass": ("bass", "sub bass", "808"),
    "brass": ("brass", "horn", "trumpet", "sax", "saxophone"),
    "melancholic": ("sad", "melancholy", "melancholic", "wistful"),
    "energetic": ("energetic", "energy", "aggressive", "intense"),
    "calm": ("calm", "quiet", "soft", "peaceful", "gentle"),
    "dark": ("dark", "brooding", "ominous"),
    "uplifting": ("happy", "uplifting", "bright", "cheerful"),
    "danceable": ("dance", "danceable", "club", "groove"),
    "instrumental": ("instrumental",),
    "acoustic": ("acoustic",),
    "electronic_prod": ("electronic", "produced", "studio"),
    "live": ("live", "concert"),
    "1960s": ("1960", "60s", "sixties"),
    "modern": ("modern", "contemporary"),
    "major_key": ("major",),
    "minor_key": ("minor",),
    "fast_tempo": ("fast", "uptempo", "up-tempo"),
    "slow_tempo": ("slow", "ballad", "down-tempo"),
}


@dataclass
class RawClip:
    """One audio file plus whatever metadata we could attach."""

    path: Path
    source: str
    track_id: str
    title: str
    genre: str
    artist_id: int
    caption: str
    extra_text: str = ""
    valence: float | None = None
    arousal: float | None = None
    start_s: float = 0.0
    tags: list[str] = field(default_factory=list)


def describe_expected_layout(root: Path | None = None) -> dict[str, Any]:
    raw = (root or ROOT) / "data" / "raw"
    return {
        "fma_audio": str(raw / "fma_small"),
        "fma_metadata": str(raw / "fma_metadata" / "tracks.csv"),
        "gtzan": str(raw / "gtzan" / "genres"),
        "musiccaps_csv": str(raw / "musiccaps" / "musiccaps-public.csv"),
        "musiccaps_audio": str(raw / "musiccaps" / "audio"),
        "deam_annotations": str(raw / "deam" / "annotations"),
        "deam_audio": str(raw / "deam" / "audio"),
        "magnatagatune": str(raw / "magnatagatune"),
        "docs": {
            "fma": FMA_URL,
            "mtt": MTT_URL,
            "gtzan": GTZAN_URL,
            "deam": DEAM_URL,
            "musiccaps": MUSICCAPS_URL,
        },
    }


def raw_root(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    return ROOT / cfg.get("paths", {}).get("raw", "data/raw")


def list_audio(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    out = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    return sorted(out)


def has_real_audio(root: Path | None = None) -> bool:
    raw = (root or ROOT) / "data" / "raw"
    if not raw.exists():
        return False
    for p in raw.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            return True
    return False


def map_genre(name: str | None) -> str:
    if not name:
        return "pop"
    key = re.sub(r"[^a-z0-9+/ -]+", "", str(name).strip().lower())
    key = key.replace("_", " ").strip()
    if key in GENRE_ALIASES:
        return GENRE_ALIASES[key]
    for alias, canon in GENRE_ALIASES.items():
        if alias in key or key in alias:
            return canon
    return "pop"


def _stable_int(text: str, modulo: int = 10_000) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % modulo


def map_tags_from_text(*parts: str, genre: str | None = None) -> np.ndarray:
    blob = " ".join(p for p in parts if p).lower()
    y = np.zeros(len(TAGS), dtype=np.float32)
    for tag, words in TAG_SYNONYMS.items():
        if any(w in blob for w in words):
            y[TAG_TO_IDX[tag]] = 1.0
    if genre and genre in GENRE_PRIOR:
        for tag, p in GENRE_PRIOR[genre]["tags"].items():
            if p >= 0.7 and tag in TAG_TO_IDX:
                y[TAG_TO_IDX[tag]] = 1.0
    if y.sum() < 3 and genre and genre in GENRE_PRIOR:
        ranked = sorted(GENRE_PRIOR[genre]["tags"].items(), key=lambda kv: -kv[1])
        for tag, _ in ranked:
            if tag in TAG_TO_IDX:
                y[TAG_TO_IDX[tag]] = 1.0
            if y.sum() >= 3:
                break
    return y


def _find_named(raw: Path, *names: str) -> Path | None:
    for name in names:
        p = raw / name
        if p.exists():
            return p
    matches = []
    for p in raw.rglob("*"):
        if p.name in names or p.name.lower() in {n.lower() for n in names}:
            matches.append(p)
    return matches[0] if matches else None


def _load_fma_tracks_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path, index_col=0, header=[0, 1], low_memory=False)


def _cell(row, col) -> str:
    try:
        v = row[col]
    except Exception:
        return ""
    if v is None:
        return ""
    try:
        if float(v) != float(v):  # NaN
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    if s.lower() in {"nan", "none", "[]", ""}:
        return ""
    return s


def _load_fma_genre_names(raw: Path) -> dict[int, str]:
    import pandas as pd

    path = raw / "fma_metadata" / "genres.csv"
    if not path.exists():
        found = _find_named(raw, "genres.csv")
        path = found if found else path
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[int, str] = {}
    for _, r in df.iterrows():
        try:
            out[int(r["genre_id"])] = str(r["title"])
        except Exception:
            continue
    return out


def _parse_int_list(text: str) -> list[int]:
    if not text:
        return []
    nums = re.findall(r"\d+", text)
    return [int(n) for n in nums]


def fma_caption_from_row(row, genre: str, title: str, genre_names: dict[int, str] | None = None) -> tuple[str, str]:
    """Build a short MusicCaps-style caption plus extra tag text from FMA metadata."""
    artist = _cell(row, ("artist", "name")) or "an unknown artist"
    album = _cell(row, ("album", "title"))
    leaf = []
    if genre_names:
        for gid in _parse_int_list(_cell(row, ("track", "genres"))):
            name = genre_names.get(gid)
            if name:
                leaf.append(name)
    leaf = leaf[:4]
    bits = [f"A {genre} recording titled {title} by {artist}"]
    if album:
        bits.append(f"from the album {album}")
    if leaf:
        bits.append("mixing " + ", ".join(leaf))
    bits.append("on Free Music Archive.")
    caption = " ".join(bits)
    extra = " ".join(
        p
        for p in (
            _cell(row, ("track", "tags")),
            _cell(row, ("album", "tags")),
            _cell(row, ("artist", "tags")),
            " ".join(leaf),
            genre,
        )
        if p
    )
    return caption, extra


def _fma_audio_path(audio_root: Path, track_id: int) -> Path | None:
    tid = f"{int(track_id):06d}"
    candidate = audio_root / tid[:3] / f"{tid}.mp3"
    if candidate.exists():
        return candidate
    # Some dumps flatten the tree or use wav.
    for ext in (".mp3", ".wav", ".flac"):
        hits = list(audio_root.rglob(f"{tid}{ext}"))
        if hits:
            return hits[0]
    return None


def collect_fma(raw: Path) -> list[RawClip]:
    audio_root = None
    for name in ("fma_small", "fma_medium", "fma_large", "fma"):
        p = raw / name
        if p.is_dir() and list_audio(p):
            audio_root = p
            break
    if audio_root is None:
        return []
    meta = _find_named(raw, "tracks.csv")
    if meta is None:
        # metadata often sits in fma_metadata/
        for p in raw.rglob("tracks.csv"):
            meta = p
            break
    clips: list[RawClip] = []
    table = _load_fma_tracks_csv(meta) if meta else None
    genre_names = _load_fma_genre_names(raw)
    audio_files = list_audio(audio_root)
    for path in audio_files:
        stem = path.stem
        if not stem.isdigit():
            continue
        tid = int(stem)
        genre = "pop"
        title = stem
        artist = _stable_int(f"fma-{tid}")
        caption = ""
        extra = ""
        if table is not None and tid in table.index:
            row = table.loc[tid]
            try:
                genre = map_genre(str(row[("track", "genre_top")]))
            except Exception:
                genre = "pop"
            try:
                title = str(row[("track", "title")])
            except Exception:
                pass
            try:
                artist = int(row[("artist", "id")])
            except Exception:
                pass
            caption, extra = fma_caption_from_row(row, genre, title, genre_names)
        clips.append(
            RawClip(
                path=path,
                source="fma",
                track_id=f"fma_{tid:06d}",
                title=title,
                genre=genre,
                artist_id=int(artist),
                caption=caption or f"A {genre} clip from Free Music Archive.",
                extra_text=extra,
            )
        )
    return clips


def collect_gtzan(raw: Path) -> list[RawClip]:
    root = None
    for cand in (raw / "gtzan" / "genres", raw / "gtzan", raw / "genres", raw / "GTZAN"):
        if cand.is_dir():
            root = cand
            break
    if root is None:
        return []
    clips = []
    for path in list_audio(root):
        genre = map_genre(path.parent.name)
        stem = path.stem
        artist = _stable_int(f"gtzan-{genre}-{stem[:-3] if len(stem) > 3 else stem}")
        clips.append(
            RawClip(
                path=path,
                source="gtzan",
                track_id=f"gtzan_{path.parent.name}_{stem}",
                title=stem,
                genre=genre,
                artist_id=artist,
                caption=f"A {genre} recording from the GTZAN genre collection ({stem}).",
                extra_text=f"{genre} {stem}",
            )
        )
    return clips


def collect_musiccaps(raw: Path) -> list[RawClip]:
    csv_path = None
    for p in raw.rglob("*.csv"):
        if "musiccap" in p.name.lower() or p.name.lower() == "musiccaps-public.csv":
            csv_path = p
            break
    if csv_path is None:
        return []
    audio_dir = csv_path.parent / "audio"
    if not audio_dir.is_dir():
        audio_dir = csv_path.parent
    index = {}
    for p in list_audio(audio_dir):
        index[p.stem] = p
        index[p.stem.replace("-", "_")] = p
    clips = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ytid = (row.get("ytid") or row.get("youtube_id") or "").strip()
            if not ytid:
                continue
            path = index.get(ytid) or index.get(ytid.replace("-", "_"))
            if path is None:
                # common: ytid_startend.wav
                hits = [p for k, p in index.items() if k.startswith(ytid)]
                path = hits[0] if hits else None
            if path is None:
                continue
            caption = (row.get("caption") or "").strip()
            aspects = row.get("aspect_list") or row.get("audioset_positive_labels") or ""
            start = float(row.get("start_s") or row.get("start") or 0.0)
            author = row.get("author_id") or ytid
            clips.append(
                RawClip(
                    path=path,
                    source="musiccaps",
                    track_id=f"mc_{ytid}",
                    title=ytid,
                    genre=map_genre(aspects),
                    artist_id=_stable_int(str(author)),
                    caption=caption or f"A music clip described as {aspects}.",
                    extra_text=str(aspects),
                    start_s=start,
                )
            )
    return clips


def collect_deam(raw: Path) -> list[RawClip]:
    ann = None
    for p in raw.rglob("*.csv"):
        name = p.name.lower()
        if "valence" in name or "arousal" in name or "static_annotations" in name or "deam" in name:
            ann = p
            break
    audio_dir = None
    for cand in (raw / "deam" / "audio", raw / "deam", raw / "MEMD_audio"):
        if cand.is_dir() and list_audio(cand):
            audio_dir = cand
            break
    if audio_dir is None:
        for p in raw.rglob("*"):
            if p.is_dir() and "deam" in p.name.lower() and list_audio(p):
                audio_dir = p
                break
    if audio_dir is None:
        return []
    by_stem = {p.stem: p for p in list_audio(audio_dir)}
    va: dict[str, tuple[float, float]] = {}
    if ann is not None:
        with ann.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = str(row.get("song_id") or row.get("id") or row.get("track_id") or "").strip()
                if not sid:
                    continue
                def _num(*keys: str) -> float | None:
                    for k in keys:
                        if k in row and row[k] not in (None, ""):
                            try:
                                return float(row[k])
                            except ValueError:
                                continue
                    return None

                v = _num("valence_mean", "valence", "mean_valence")
                a = _num("arousal_mean", "arousal", "mean_arousal")
                if v is None or a is None:
                    continue
                # DEAM valence/arousal are often 1–9; map to [0,1].
                if v > 1.5 or a > 1.5:
                    v = (v - 1.0) / 8.0
                    a = (a - 1.0) / 8.0
                va[sid] = (float(np.clip(v, 0, 1)), float(np.clip(a, 0, 1)))
    clips = []
    for stem, path in by_stem.items():
        v, a = va.get(stem, (None, None))
        if v is None:
            # song_id sometimes is zero-padded
            for key, pair in va.items():
                if key.lstrip("0") == stem.lstrip("0"):
                    v, a = pair
                    break
        clips.append(
            RawClip(
                path=path,
                source="deam",
                track_id=f"deam_{stem}",
                title=stem,
                genre="pop",
                artist_id=_stable_int(f"deam-{stem}"),
                caption=f"A music recording annotated for valence and arousal (DEAM {stem}).",
                extra_text="emotion valence arousal",
                valence=v,
                arousal=a,
            )
        )
    return clips


def collect_mtt(raw: Path) -> list[RawClip]:
    csv_path = None
    for p in raw.rglob("*.csv"):
        name = p.name.lower()
        if name in {"annotations_final.csv", "annotations.csv"} or "annotation" in name:
            csv_path = p
            break
    if csv_path is None:
        for p in raw.rglob("annotations_final.csv"):
            csv_path = p
            break
    if csv_path is None:
        return []
    root = csv_path.parent
    clips = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tag_cols = [c for c in (reader.fieldnames or []) if c not in {"clip_id", "mp3_path", "url"}]
        for row in reader:
            rel = row.get("mp3_path") or row.get("path") or ""
            path = (root / rel).resolve() if rel else None
            if path is None or not path.exists():
                clip_id = str(row.get("clip_id") or "")
                hits = list(root.rglob(f"*{clip_id}*"))
                path = hits[0] if hits else None
            if path is None or not path.exists():
                continue
            active = [c for c in tag_cols if str(row.get(c, "0")).strip() in {"1", "1.0", "True", "true"}]
            genre = map_genre(" ".join(active))
            caption = f"A MagnaTagATune clip tagged {', '.join(active[:8]) or 'with mixed tags'}."
            clips.append(
                RawClip(
                    path=path,
                    source="magnatagatune",
                    track_id=f"mtt_{row.get('clip_id', path.stem)}",
                    title=str(row.get("clip_id") or path.stem),
                    genre=genre,
                    artist_id=_stable_int(str(row.get("clip_id") or path.stem)),
                    caption=caption,
                    extra_text=" ".join(active),
                    tags=active,
                )
            )
    return clips


def collect_loose_audio(raw: Path, claimed: set[Path]) -> list[RawClip]:
    clips = []
    for path in list_audio(raw):
        if path.resolve() in claimed:
            continue
        genre = map_genre(path.parent.name)
        clips.append(
            RawClip(
                path=path,
                source="raw",
                track_id=f"raw_{path.stem}",
                title=path.stem,
                genre=genre,
                artist_id=_stable_int(str(path.parent)),
                caption=f"A {genre} recording ({path.stem}).",
                extra_text=path.stem.replace("_", " ").replace("-", " "),
            )
        )
    return clips


def discover_clips(raw: Path) -> list[RawClip]:
    buckets = [
        collect_fma(raw),
        collect_gtzan(raw),
        collect_musiccaps(raw),
        collect_deam(raw),
        collect_mtt(raw),
    ]
    clips: list[RawClip] = []
    claimed: set[Path] = set()
    for group in buckets:
        for c in group:
            key = c.path.resolve()
            if key in claimed:
                continue
            claimed.add(key)
            clips.append(c)
    clips.extend(collect_loose_audio(raw, claimed))
    return clips


def _downsample2d(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = arr.shape
    ys = np.linspace(0, h, out_h, endpoint=False).astype(int)
    xs = np.linspace(0, w, out_w, endpoint=False).astype(int)
    return arr[ys][:, xs].astype(np.float32)


def _crop(y: np.ndarray, sr: int, duration: float, start_s: float = 0.0) -> np.ndarray:
    start = max(0, int(start_s * sr))
    if start >= len(y):
        start = 0
    n = int(duration * sr)
    chunk = y[start : start + n]
    if len(chunk) < sr:  # < 1 s
        return chunk
    if len(chunk) < n:
        pad = np.zeros(n, dtype=np.float32)
        pad[: len(chunk)] = chunk
        return pad
    return chunk.astype(np.float32)


def featurize_clip(clip: RawClip, cfg: dict[str, Any]) -> dict[str, Any] | None:
    sr = int(cfg["sample_rate"])
    duration = float(cfg["data"].get("real_duration_seconds", cfg.get("duration_seconds", 8)))
    try:
        y, sr = load_mono(str(clip.path), sr=sr)
    except Exception:
        return None
    y = _crop(y, sr, duration, clip.start_s)
    if len(y) < int(0.5 * sr):
        return None
    bundle = extract_bundle(y, sr)
    graphs = build_graphs(bundle)
    tags = map_tags_from_text(clip.caption, clip.extra_text, " ".join(clip.tags), genre=clip.genre)
    if clip.valence is None or clip.arousal is None:
        prior = GENRE_PRIOR.get(clip.genre, GENRE_PRIOR["pop"])
        valence = float(np.mean(prior["valence"]))
        arousal = float(np.mean(prior["arousal"]))
    else:
        valence, arousal = float(clip.valence), float(clip.arousal)
    genre = clip.genre if clip.genre in GENRE_TO_IDX else "pop"
    return {
        "id": clip.track_id,
        "title": clip.title,
        "genre": genre,
        "genre_idx": GENRE_TO_IDX[genre],
        "artist_id": int(clip.artist_id),
        "caption": clip.caption,
        "tags": tags,
        "tag_names": [TAGS[j] for j, v in enumerate(tags) if v > 0],
        "valence": valence,
        "arousal": arousal,
        "tempo": float(bundle.tempo),
        "chord_seq": graphs["chord"].node_labels,
        "segment_graph": graphs["segment"],
        "chord_graph": graphs["chord"],
        "mel_small": _downsample2d(bundle.log_mel, 64, 32),
        "chroma_small": _downsample2d(bundle.chroma, 12, 16),
        "n_mels_frames": int(bundle.log_mel.shape[1]),
        "source": clip.source,
        "audio_path": str(clip.path),
    }


def build_real_corpus(cfg: dict[str, Any] | None = None, progress: bool = True) -> dict[str, Any]:
    """Scan data/raw, featurize audio, return the same structure as synthetic.build_corpus."""
    from src.synthetic import artist_splits

    cfg = cfg or load_config()
    raw = raw_root(cfg)
    clips = discover_clips(raw)
    if not clips:
        layout = json.dumps(describe_expected_layout(), indent=2)
        raise FileNotFoundError(
            "No audio found under data/raw/.\n"
            "Place FMA / GTZAN / MusicCaps / DEAM / MagnaTagATune files using this layout:\n"
            f"{layout}"
        )
    limit = int(cfg["data"].get("max_real_tracks") or 0)
    if limit > 0:
        clips = clips[:limit]
    tracks = []
    iterator = clips
    if progress:
        iterator = tqdm(clips, desc="real audio → graphs")
    skipped = 0
    for clip in iterator:
        rec = featurize_clip(clip, cfg)
        if rec is None:
            skipped += 1
            continue
        tracks.append(rec)
    if len(tracks) < 8:
        raise RuntimeError(
            f"Only {len(tracks)} clips could be decoded (skipped {skipped}). "
            "Need at least 8 readable audio files under data/raw/."
        )
    splits = artist_splits(tracks, cfg)
    sources = sorted({t.get("source", "raw") for t in tracks})
    return {
        "tracks": tracks,
        "splits": splits,
        "feat_dim": node_feature_dim(),
        "source": ",".join(sources),
        "n_skipped": skipped,
    }


def inventory(root: Path | None = None) -> dict[str, Any]:
    raw = (root or ROOT) / "data" / "raw"
    clips = discover_clips(raw) if raw.exists() else []
    by_src: dict[str, int] = {}
    for c in clips:
        by_src[c.source] = by_src.get(c.source, 0) + 1
    return {
        "raw": str(raw),
        "n_clips": len(clips),
        "by_source": by_src,
        "audio_files": len(list_audio(raw)) if raw.exists() else 0,
        "layout": describe_expected_layout(root or ROOT),
    }


if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2))
