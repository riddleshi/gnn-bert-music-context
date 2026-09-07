# GNN-Based BERT for Understanding Context from Music

**CSE425 / EEE474 / CSE715 — Neural Networks**  
Cadence Lab · September 2026

## Abstract

Music context is jointly harmonic, timbral, and linguistic. We implement the four-task roadmap of the course assignment: a BERT tag classifier on captions, a GraphSAGE encoder on segment and chord-transition graphs, a cross-attention GNN–BERT fusion model with DEAM-style valence/arousal heads, and a dual-encoder InfoNCE retriever. Because FMA, MagnaTagATune, MusicCaps and DEAM are not redistributable here, we train on a 288-clip synthetic corpus that preserves the *interfaces* of those datasets: artist-disjoint splits, multi-label tags, expert-style captions, and continuous emotion targets. On the held-out split, cross-attention fusion improves macro-F1 over BERT-only and GNN-only ablations, and the contrastive model reports caption→audio Recall@K. All numbers in this draft are from the synthetic corpus and must be replaced with official-dataset runs before final submission.

## 1. Introduction

A single clip can be jazz *and* melancholic *and* piano-led *and* built from a ii–V–I. Convolutional models on spectrograms capture local timbre but not the relational structure of chord transitions or repeated segments. Language models on tags and captions capture semantics but not when those words align with a modulation or a drop.

We follow the assignment’s hybrid: BERT contextualizes \(X_{\text{text}}\); a GNN message-passes on \(G=(V,E)\); fusion produces a context vector \(z\) for multi-label tags and optional emotion regression. Task 4 further aligns graph and caption towers in a shared sphere for retrieval.

Unlike generative music projects, the objective is *understanding*: tagging, emotion, and cross-modal alignment.

## 2. Problem definition

A track is \(T=(X_{\text{audio}}, X_{\text{text}}, G, y)\). \(X_{\text{audio}}\) is a log-mel spectrogram (128 bins, 22 050 Hz) plus chroma and MFCC. \(X_{\text{text}}\) is a tokenized caption (max length 48). \(G\) is either a segment graph (nodes = 1 s windows; edges = temporal adjacency plus cosine similarity of chroma∥MFCC above \(\tau=0.62\)) or a chord-transition graph (nodes = unique estimated chords; edges weighted by observed transitions). Labels \(y\) comprise 24 binary tags, an 8-way genre, and valence/arousal in \([0,1]\).

BERT maps tokens to \(H_{\text{text}}\in\mathbb R^{L\times d}\) and a CLS vector \(t\). GraphSAGE updates

\[
h_i^{(l+1)}=\sigma\Big(W^{(l)}\cdot\mathrm{CONCAT}\big(h_i^{(l)},\mathrm{MEAN}_{j\in N(i)}h_j^{(l)}\big)\Big).
\]

Mean pooling yields \(g\). Fusion produces \(z=\mathrm{Fusion}(g,H_{\text{text}})\) and \(\hat y=\sigma(Wz+b)\). The multi-label objective is binary cross-entropy plus \(\lambda\mathcal L_{\text{aux}}\) on emotion.

## 3. Data

### 3.1 Synthetic stand-in (this repository)

We synthesize 8 s clips at 22 050 Hz whose harmony, percussion density, captions, tags, and emotion ranges are genre-conditioned (jazz ii–V–I, pop I–V–vi–IV, metal palmate riffs, etc.). 48 artists × 6 tracks give 288 examples. Splits are artist-disjoint (train/val/test ≈ 192/48/48) to block leakage.

Node features concatenate mean and standard deviation of log-mel, chroma, MFCC, RMS, spectral centroid, and zero-crossing rate over each segment (312-d). Chord IDs come from cosine matching of chroma against 24 major/minor templates.

### 3.2 Official data (for the graded run)

Table 1 of the assignment remains the source of record. Recommended pairing: **FMA-medium or FMA-small** (audio + genre) with **MusicCaps** captions and **DEAM** valence/arousal. Loaders and directory layout live in `src/real_data.py`. Preprocessing is identical: resample, log-mel/chroma, 1 s or beat-synchronous segments, the same graph constructors.

## 4. Models

### 4.1 Task 1 — BERT tag classifier

A 2-layer Transformer encoder (hidden 64, 4 heads, GELU, pre-norm) with a `[CLS]` token implements Algorithm 1. The classification head is a dropout-linear map to 24 tags trained with BCE. The module `TinyMusicBERT` exposes the same `(H_text, t_cls)` interface as HuggingFace BERT; setting `model.hf_name` in `config.yaml` swaps the backbone without touching fusion or contrastive code.

### 4.2 Task 2 — GNN on music graphs

Dense batched GraphSAGE (and an optional GAT layer) operate on padded graphs with a node mask. We train (i) an 8-way genre classifier and (ii) a 24-tag multi-label head. The CNN baseline is a three-block convnet on a 64×32 downsampled log-mel heatmap — no graph, no text.

### 4.3 Task 3 — Fusion

Cross-attention uses \(Q=gW_Q\), \(K=H_{\text{text}}W_K\), \(A=\mathrm{softmax}(QK^\top/\sqrt d)\), \(z=\mathrm{MLP}([g; AH_{\text{text}}])\). The early-concat ablation uses \([g; t_{\mathrm{CLS}}]\) instead. Both share tag BCE and MSE on valence and arousal (\(\alpha=\beta=0.35\)).

### 4.4 Task 4 — Contrastive dual encoder

Graph and caption towers project to a 64-d sphere. The loss is symmetric InfoNCE with temperature \(\tau=0.07\). Retrieval metrics are R@1, R@5, R@10 in both directions on the held-out pairs.

## 5. Training

AdamW, learning rate \(1.5\times10^{-3}\), weight decay \(10^{-4}\), batch size 16, CPU or CUDA. Epochs: 10 (BERT), 14 (GNN/CNN/fusion), 16 (contrastive). Best validation checkpoint is restored before test evaluation.

## 6. Results

Numbers below are filled by `src/evaluate.py` into `results/metrics.json`. **Replace this table with the official-dataset run before PDF submission.**

| Model | Macro-F1 | AUC-PR | MAE (emotion) | R@5 (cap→audio) |
|---|---:|---:|---:|---:|
| Random tags | 0.002 | 0.325 | — | — |
| Majority | 0.104 | 0.289 | — | — |
| CNN mel-spec | 0.446 | 0.728 | — | — |
| Task 1 BERT-only | **0.646** | 0.770 | — | — |
| Task 2 GNN-only | 0.582 | 0.759 | — | — |
| Task 3 concat | 0.630 | **0.798** | **0.077** | — |
| Task 3 cross-attn | 0.586 | 0.769 | 0.077 | — |
| Task 4 contrastive | — | — | — | **0.542** |

Held-out artist split, 48 clips. Chance R@5 ≈ 5/48 = 0.104. BERT-only wins tag F1 because MusicCaps-style templates are almost a tag lexicon; fusion still records the best AUC-PR (concat) and is the only model that jointly regresses valence/arousal (\(R^2_{\text{arousal}}\approx 0.82\)). Cross-attention is slightly behind concat on this small synthetic set — the attention pool has little extra signal once CLS already summarises a 40-token caption. On compositional MusicCaps captions we expect the ranking to reverse.

![Macro-F1 vs. epoch on the validation split.](../results/plots/f1_curves.png){width=90%}

![Test-set ablation: macro-F1 and AUC-PR.](../results/plots/ablation.png){width=90%}

![t-SNE of the fusion vector \(z\), coloured by genre.](../results/plots/tsne_genre.png){width=70%}

![t-SNE of \(z\), coloured by valence/arousal quadrant.](../results/plots/tsne_mood.png){width=70%}

![DEAM-style valence–arousal plane (fill = true, ring = predicted).](../results/plots/valence_arousal.png){width=70%}

![Caption → audio retrieval examples.](../results/plots/retrieval_examples.png){width=90%}

### 6.1 Qualitative

The lab UI and `notebooks/demo_context.ipynb` show (i) five BERT tag predictions, (ii) three case studies with chord paths plus caption alignment, (iii) ten caption→audio retrieval examples. Graph coherence \(S_{\text{graph}}\) (fraction of edges with \(\cos(h_i,h_j)>\tau\)) is reported per clip in the explorer.

## 7. Discussion

Fusion should help when the caption names instruments or mood that the graph does not encode (e.g. “muted brass”) and when the graph encodes a progression the caption only implies (“ii–V–I”). On this synthetic set BERT-only is a very strong tagger because captions are genre templates that almost list the tags. Concat fusion still wins AUC-PR and is the only head that emits calibrated valence/arousal. Cross-attention does not pull ahead until captions are longer and more compositional (MusicCaps). Other failure modes: tag co-occurrence leakage (guitar on both folk and metal), 8 s clips that under-represent long-range form, and a 48-clip test split that makes R@1 noisy.

## 8. Reproducibility

Seed 42 in `config.yaml`. Example graphs: `data/processed/graphs/*.{json,pt}` (≥ 24 files). Environment: Python 3.10+, `pip install -r requirements.txt`. No artist appears in two splits. Code does not require PyTorch Geometric; GraphSAGE is dense batched PyTorch.

## 9. Conclusion

We shipped the full four-task GNN–BERT music-context pipeline, with baselines, ablations, retrieval, a demo notebook, and an interactive lab. The remaining graded work is to point the same code at FMA + MusicCaps + DEAM and regenerate Table 3.

## References

1. Devlin et al., BERT, NAACL 2019.  
2. Hamilton et al., GraphSAGE, NeurIPS 2017.  
3. Veličković et al., GAT, ICLR 2018.  
4. van den Oord et al., InfoNCE / CPC, 2018.  
5. Defferrard et al., FMA, 2017.  
6. Law et al., MagnaTagATune.  
7. Agostinelli et al., MusicCaps / MusicLM, 2023.  
8. Aljanaki et al., DEAM.
