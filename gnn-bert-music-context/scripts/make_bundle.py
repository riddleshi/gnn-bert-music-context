#!/usr/bin/env python3
"""Package the project into a submission zip and publish it to the lab.

    python scripts/make_bundle.py

Raw FMA audio, node_modules, and build caches are excluded; the processed
corpus, saved graphs, checkpoints, plots, and the typeset report are kept so a
grader can rerun training without the 7 GB download.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from src.config import ROOT

BUNDLE = "gnn-bert-music-context"

SKIP_DIRS = {
    ".git",
    ".next",
    ".pytest_cache",
    ".ipynb_checkpoints",
    "__pycache__",
    "node_modules",
    "out",
    ".venv",
    "venv",
}
SKIP_PATHS = {Path("data/raw"), Path("uploads")}
SKIP_SUFFIXES = {".zip", ".pyc", ".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".toc"}


def _keep(rel: Path) -> bool:
    if any(part in SKIP_DIRS for part in rel.parts):
        return False
    if any(rel == p or p in rel.parents for p in SKIP_PATHS):
        return False
    return rel.suffix not in SKIP_SUFFIXES


def main() -> None:
    out = ROOT / f"{BUNDLE}.zip"
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and _keep(p.relative_to(ROOT)))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            zf.write(path, Path(BUNDLE) / path.relative_to(ROOT))

    public = ROOT / "web" / "public" / f"{BUNDLE}.zip"
    shutil.copy(out, public)
    mb = out.stat().st_size / 1e6
    print(f"wrote {out.relative_to(ROOT)} ({len(files)} files, {mb:.1f} MB)")
    print(f"lab copy → {public.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
