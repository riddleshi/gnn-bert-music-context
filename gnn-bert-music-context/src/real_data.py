"""Optional loaders for official datasets (FMA, GTZAN, MusicCaps, DEAM, MTT).

The default pipeline trains on the synthetic corpus in `src/synthetic.py`.
When you download a real dump into data/raw/, point these helpers at it
and emit the same cached record schema (`seg_x`, `seg_adj`, `caption`, `tags`, …)
so `src.train` does not change.
"""

from __future__ import annotations

from pathlib import Path

FMA_URL = "https://github.com/mdeff/fma"
MTT_URL = "https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset"
GTZAN_URL = "https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification"
DEAM_URL = "https://cvml.unige.ch/databases/DEAM/"
MUSICCAPS_URL = "https://www.kaggle.com/datasets/googleai/musiccaps"
MSD_URL = "http://millionsongdataset.com/"
LMD_URL = "https://colinraffel.com/projects/lmd/"


def describe_expected_layout(root: Path) -> dict[str, str]:
    raw = root / "data" / "raw"
    return {
        "fma_small": str(raw / "fma_small"),
        "fma_metadata": str(raw / "fma_metadata"),
        "gtzan": str(raw / "gtzan" / "genres"),
        "musiccaps_csv": str(raw / "musiccaps" / "musiccaps-public.csv"),
        "deam_annotations": str(raw / "deam" / "annotations"),
        "magnatagatune": str(raw / "magnatagatune"),
        "docs": {
            "fma": FMA_URL,
            "mtt": MTT_URL,
            "gtzan": GTZAN_URL,
            "deam": DEAM_URL,
            "musiccaps": MUSICCAPS_URL,
            "msd": MSD_URL,
            "lmd": LMD_URL,
        },
    }


def has_fma_small(root: Path) -> bool:
    return (root / "data" / "raw" / "fma_small").exists()
