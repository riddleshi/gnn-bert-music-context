#!/usr/bin/env python3
"""Generate the synthetic corpus and persist graphs / splits."""

from src.config import load_config
from src.synthetic import build_corpus, persist_corpus


def main() -> None:
    cfg = load_config()
    corpus = build_corpus(cfg, progress=True)
    persist_corpus(corpus, cfg)
    print(f"wrote {len(corpus['tracks'])} tracks, feat_dim={corpus['feat_dim']}")
    print("splits", {k: len(v) for k, v in corpus["splits"].items()})


if __name__ == "__main__":
    main()
