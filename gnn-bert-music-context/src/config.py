"""Load project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    return cfg


def resolve_device(cfg: dict[str, Any]) -> str:
    import torch

    requested = str(cfg.get("train", {}).get("device", "auto")).lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested
