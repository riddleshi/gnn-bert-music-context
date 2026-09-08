#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python scripts/build_corpus.py
python -m src.train --task all
python -m src.evaluate
python -m pytest tests -q
