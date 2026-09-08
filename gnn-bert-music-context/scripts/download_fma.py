#!/usr/bin/env python3
"""Download official FMA dumps and extract a genre-balanced subset.

Audio is written under data/raw/ (gitignored). Checksums match
https://github.com/mdeff/fma — files are hosted at
https://os.unil.cloud.switch.ch/fma/

    PYTHONPATH=. python scripts/download_fma.py              # metadata + fma_small subset
    PYTHONPATH=. python scripts/download_fma.py --subset 512
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from src.config import ROOT

HOST = "https://os.unil.cloud.switch.ch/fma"
SHA1 = {
    "fma_metadata.zip": "f0df49ffe5f2a6008d7dc17a268956ba1d9b5bd3",
    "fma_small.zip": "ade154f733639d52e35e32f5593efe5be76c6d70",
}
MUSICCAPS = (
    "https://raw.githubusercontent.com/google-research/seanet/main/"
    "audiolm/data/musiccaps-public.csv"
)
FMA_SMALL_GENRES = [
    "Hip-Hop",
    "Pop",
    "Folk",
    "Experimental",
    "Rock",
    "International",
    "Instrumental",
    "Electronic",
]


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(name: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and _sha1(dest) == SHA1[name]:
        print(f"ok {dest.name}")
        return dest
    url = f"{HOST}/{name}"
    print(f"downloading {url}")
    urlretrieve(url, dest)
    digest = _sha1(dest)
    if digest != SHA1[name]:
        raise SystemExit(f"sha1 mismatch for {name}: {digest}")
    print(f"verified {name}")
    return dest


def _extract_metadata(zpath: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(out.parent)
    print("extracted metadata →", out)


def _subset_ids(meta_csv: Path, n: int, seed: int = 42) -> list[int]:
    import numpy as np
    import pandas as pd

    tracks = pd.read_csv(meta_csv, index_col=0, header=[0, 1], low_memory=False)
    small = tracks[tracks[("set", "subset")] == "small"]
    per = max(1, n // len(FMA_SMALL_GENRES))
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for g in FMA_SMALL_GENRES:
        pool = small[small[("track", "genre_top")] == g].index.to_numpy()
        take = min(per, len(pool))
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    return sorted(int(i) for i in chosen[:n])


def _extract_subset(zpath: Path, ids: list[int], audio_root: Path) -> None:
    wanted = {f"{i:06d}.mp3" for i in ids}
    audio_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if name not in wanted:
                continue
            tid = name.replace(".mp3", "")
            dest = audio_root / tid[:3] / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                copied += 1
                continue
            with zf.open(info) as src, dest.open("wb") as out:
                out.write(src.read())
            copied += 1
    print(f"extracted {copied} mp3s → {audio_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=512)
    parser.add_argument("--cache", default="/tmp/fma-dl")
    args = parser.parse_args()
    cache = Path(args.cache)
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    meta_zip = _fetch("fma_metadata.zip", cache / "fma_metadata.zip")
    meta_dir = raw / "fma_metadata"
    if not (meta_dir / "tracks.csv").exists():
        _extract_metadata(meta_zip, meta_dir)

    ids = _subset_ids(meta_dir / "tracks.csv", args.subset)
    (raw / "fma_small_subset_ids.txt").write_text("\n".join(f"{i:06d}" for i in ids) + "\n")

    small_zip = _fetch("fma_small.zip", cache / "fma_small.zip")
    _extract_subset(small_zip, ids, raw / "fma_small")

    caps = raw / "musiccaps" / "musiccaps-public.csv"
    if not caps.exists():
        caps.parent.mkdir(parents=True, exist_ok=True)
        print("downloading MusicCaps metadata CSV (no YouTube audio)")
        try:
            urlretrieve(MUSICCAPS, caps)
        except Exception as exc:
            print("MusicCaps CSV skipped:", exc)

    print("done. next: PYTHONPATH=. python scripts/build_corpus.py --source real")


if __name__ == "__main__":
    main()
