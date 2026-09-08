# Cadence Lab — GNN-Based BERT for Music Context

Course project for **Neural Networks (CSE425 / EEE474 / CSE715)**. A hybrid **TinyBERT + GraphSAGE** system that treats a music clip as the tuple

\[
T = (X_{\text{audio}}, X_{\text{text}}, G, y)
\]

and predicts multi-label tags, genre, valence/arousal, and caption↔audio retrieval. This is an *understanding* model, not a generator.

The interactive lab UI is the fastest way to inspect graphs, ablations, and live fusion inference.

## What is implemented

| Task | Model | Code |
|---|---|---|
| 1 Easy | BERT multi-label tag classifier on captions | `src/bert_encoder.py`, `src/train.py` |
| 2 Medium | GraphSAGE (and GAT) on segment / chord graphs vs CNN mel baseline | `src/graph_builder.py`, `src/gnn_model.py`, `src/cnn_baseline.py` |
| 3 Hard | Cross-attention GNN–BERT fusion + early-concat ablation + emotion heads | `src/fusion_model.py` |
| 4 Advanced | Dual-encoder InfoNCE, R@1/5/10 | `src/contrastive.py` |

Baselines: random tags, majority-class, CNN on log-mel, BERT-only, GNN-only.

## Repository layout

```
README.md
requirements.txt
config.yaml
src/                  # audio, graphs, BERT, GNN, fusion, train, eval
api/server.py         # FastAPI live inference (port 43181)
web/                  # Next.js lab UI (port 43180)
notebooks/eda.ipynb
notebooks/demo_context.ipynb
data/processed/       # cached corpus + ≥20 example .pt/.json graphs
data/splits/splits.json
results/              # metrics.json, plots, checkpoints
report/final_report.md
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.

python scripts/build_corpus.py          # real dumps in data/raw/, else synthetic stand-in
python -m src.train --task all
python -m src.evaluate

# API + UI
python -m uvicorn api.server:app --host 0.0.0.0 --port 43181
cd web && npm install && npm run dev
```

Open [http://127.0.0.1:43180](http://127.0.0.1:43180).

Unit tests: `PYTHONPATH=. python -m pytest tests -q`.

## Data

This run uses a **genre-balanced FMA-small subset** (512 requested, 511 decoded mp3s) plus the **MusicCaps public CSV** (captions only — no YouTube audio). DEAM, MagnaTagATune, and GTZAN audio are not on disk; valence/arousal fall back to FMA-genre priors. Audio lives under `data/raw/` and is gitignored.

```bash
PYTHONPATH=. python scripts/download_fma.py          # official FMA-small + metadata (SHA1-checked)
PYTHONPATH=. python -m src.real_data                 # inventory of data/raw
PYTHONPATH=. python scripts/build_corpus.py          # uses real audio if present
PYTHONPATH=. python scripts/build_corpus.py --source synthetic   # force 288-clip stand-in
```

Official dumps (FMA, MagnaTagATune, MusicCaps audio, DEAM, GTZAN) are large and are **not** vendored. If `data/raw/` is empty, `src/synthetic.py` still builds a 288-track artist-split corpus so every script runs offline.

Preprocessing matches the brief: 22 050 Hz, 128-bin log-mel, 12-bin chroma, 1 s segments, temporal edges plus cosine(chroma∥MFCC) > τ, chord-transition graphs via 24 major/minor templates.

Splits are **artist-disjoint** (`data/splits/splits.json`).

## Models (equations as implemented)

**Task 1.** \( t = \mathrm{BERT}_{\mathrm{CLS}}(X_{\text{text}}),\ \hat y_k = \sigma(w_k^\top t + b_k) \), BCE over 24 MagnaTagATune-style tags.

**Task 2.** GraphSAGE
\( h_i^{(l+1)} = \sigma\big(W^{(l)}\cdot \mathrm{CONCAT}(h_i^{(l)}, \mathrm{MEAN}_{j\in N(i)} h_j^{(l)})\big) \), mean-pool readout, genre CE + tag BCE.

**Task 3.** \( A = \mathrm{softmax}(QK^\top/\sqrt d),\ Q=gW_Q,\ K=H_{\text{text}}W_K \), \( z=\mathrm{CONCAT}(g, AH_{\text{text}}) \),
\( \mathcal L = \mathcal L_{\text{tags}} + \alpha\|v-\hat v\|_2^2 + \beta\|a-\hat a\|_2^2 \).

**Task 4.** Symmetric InfoNCE on \(\ell_2\)-normalized graph and caption towers.

The default text encoder is a 2-layer, 64-d Transformer (`TinyMusicBERT`) so CPU training finishes in minutes. Set `model.hf_name: bert-base-uncased` in `config.yaml` after installing `transformers` to swap in HuggingFace BERT; the rest of the pipeline is unchanged.

## Demo notebook

`notebooks/demo_context.ipynb` loads one held-out clip, draws its graphs, and runs fusion inference.

## Report

IEEE conference paper (`IEEEtran`, two-column, 6–10 pages): [`report/final_report.pdf`](report/final_report.pdf) (LaTeX: [`report/final_report.tex`](report/final_report.tex)). Rebuild and publish into the lab Paper tab with:

```bash
PYTHONPATH=. python scripts/generate_report.py
```

## Slides and speaker script

Twelve-slide course talk plus an **8–10 minute** script:

- PowerPoint: [`report/cadence_lab_slides.pptx`](report/cadence_lab_slides.pptx) (download from the lab: `/cadence-lab-slides.pptx`). Rebuild with `python scripts/generate_pptx.py`.
- Web deck: [http://127.0.0.1:43180/slides](http://127.0.0.1:43180/slides) (arrow keys, **N** for notes)
- Timed script: [http://127.0.0.1:43180/slides/script](http://127.0.0.1:43180/slides/script) and [`report/presentation_script.md`](report/presentation_script.md)
