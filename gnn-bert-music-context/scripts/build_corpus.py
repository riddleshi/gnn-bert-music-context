#!/usr/bin/env python3
"""Build the training cache.

If audio exists under data/raw/, featurize those official dumps.
Otherwise synthesize the offline stand-in corpus.

    PYTHONPATH=. python scripts/build_corpus.py
    PYTHONPATH=. python scripts/build_corpus.py --source real
    PYTHONPATH=. python scripts/build_corpus.py --source synthetic
    PYTHONPATH=. python -m src.real_data   # print what data/raw currently contains
"""

from __future__ import annotations

import argparse

from src.config import load_config
from src.real_data import build_real_corpus, has_real_audio, inventory
from src.synthetic import build_corpus, persist_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=("auto", "real", "synthetic"),
        default=None,
        help="auto (default) uses data/raw when audio is present",
    )
    args = parser.parse_args()
    cfg = load_config()
    source = args.source or str(cfg.get("data", {}).get("source", "auto"))
    use_real = source == "real" or (source == "auto" and has_real_audio())
    if source == "real" and not has_real_audio():
        print(inventory())
        raise SystemExit("data/raw/ has no audio files. Drop FMA/GTZAN/MusicCaps/DEAM there and retry.")
    if use_real:
        print("building corpus from data/raw/")
        print("inventory:", inventory()["by_source"])
        corpus = build_real_corpus(cfg, progress=True)
        persist_corpus(corpus, cfg)
        print(
            f"wrote {len(corpus['tracks'])} real tracks "
            f"from {corpus.get('source')} (skipped {corpus.get('n_skipped', 0)}), "
            f"feat_dim={corpus['feat_dim']}"
        )
    else:
        print("no audio in data/raw/ — synthesizing the offline stand-in corpus")
        corpus = build_corpus(cfg, progress=True)
        persist_corpus(corpus, cfg)
        print(f"wrote {len(corpus['tracks'])} synthetic tracks, feat_dim={corpus['feat_dim']}")
    print("splits", {k: len(v) for k, v in corpus["splits"].items()})


if __name__ == "__main__":
    main()
