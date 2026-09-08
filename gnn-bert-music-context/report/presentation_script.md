# Speaker script — GNN-Based BERT for Understanding Context from Music

**CSE425 / EEE474 / CSE715 — Neural Networks**  
Cadence Lab · 12 slides · **8–10 minutes**

Open the deck at `/slides` or download `report/cadence_lab_slides.pptx`. Press **N** on the web deck, or use PowerPoint Presenter View for the same notes.
If you run long on Data or Reading the table, skip the three track titles and jump to takeaways.

Total spoken length: **1178 words** (~8.4 min at 140 wpm, ~9.1 min with pauses on the numbers). Slide clock sums to **10:00**.

---

### 0:00–0:45 · Slide 1 · Course presentation

**GNN-Based BERT for Understanding Context from Music**

This is Cadence Lab’s course project for Neural Networks: GNN-based BERT for understanding context from music. The assignment is not to generate audio. It is to read a clip as structure plus language, then predict tags, genre, mood, and retrieval. Everything I will show was trained on official Free Music Archive small audio: five hundred eleven thirty-second clips, artist-disjoint splits. I will be honest about where the labels leak from metadata, because that is the actual finding, not a leaderboard claim. After the numbers I will point you to the lab, the saved graphs, and the seven-page IEEE report.

---

### 0:45–1:35 · Slide 2 · Motivation

**A clip is harmonic, timbral, and linguistic at once**

A recording can be folk, acoustic, guitar-led, and built from a short chord loop at the same time. A spectrogram CNN hears local timbre. It does not see that bar three repeats bar one, or that A minor goes to C major. A language model on a caption hears “a folk recording titled Arabesque by Ed Askew.” It does not know when those words line up with a modulation. The brief therefore asks for a hybrid: BERT on text, a graph net on a structure graph from audio, a fusion stage, and a contrastive retriever. Five beats in this talk: why, representation, the four models, held-out numbers, then what those numbers actually mean.

---

### 1:35–2:20 · Slide 3 · Assignment

**Four tasks, one pipeline**

Task one is easy on paper: a BERT tag classifier on captions, twenty-four MagnaTagATune-style labels, binary cross-entropy. Task two is the graph: GraphSAGE, and GAT, on a segment-similarity graph and a chord-transition graph, against a CNN mel baseline. Task three fuses the graph readout with token states, concat versus cross-attention, plus valence and arousal heads. Task four is a dual encoder with InfoNCE. We also ship majority and random tag baselines so the table is complete.

---

### 2:20–3:00 · Slide 4 · Representation

**A track is the tuple T = (audio, text, G, y)**

Audio is a one-twenty-eight-bin log-mel at twenty-two thousand fifty Hertz, hop five twelve, plus chroma, MFCCs, RMS, centroid, and zero-crossing rate, all z-scored per clip. Text is a whitespace-tokenized caption of at most forty-eight tokens with CLS and SEP. G is either a thirty-node segment graph or a chord-transition graph. Labels y are twenty-four binary tags, eight-way genre, and valence-arousal in zero to one. Jazz and metal heads exist in the classifier but FMA-small has no such top genres, so those rows stay empty.

---

### 3:00–3:55 · Slide 5 · Corpus

**Official FMA-small, not a synthetic stand-in**

We fetched FMA metadata and FMA-small from the official SWITCH mirror and SHA1-checked both zips. Seed forty-two, sixty-four clips from each of eight top genres. One mp3 failed to decode, leaving five hundred eleven tracks from three hundred eighty-six artists. Splits are artist-disjoint: three hundred twenty-seven, ninety-four, ninety. After mapping, folk and electronic have one hundred twenty-eight clips each; hip-hop has sixty-three. Experimental maps to electronic, instrumental to classical, international to folk. Captions are templates from title, artist, album, and leaf genres. MusicCaps CSV is on disk without YouTube audio. DEAM is gated, so valence and arousal are genre priors. Tags are keyword-mapped from those captions. That last fact will explain BERT’s F1.

---

### 3:55–4:50 · Slide 6 · Task 2

**Two graphs, both deterministic functions of the spectrogram**

The segment graph has thirty windows of one second. Each node concatenates mean and standard deviation of the audio streams: three hundred twelve dimensions. Adjacent windows get a temporal edge. If chroma concatenated with MFCC cosine exceeds tau 0.62, we add a similarity edge. That is a self-similarity matrix with a backbone. The chord graph matches frames to twenty-four major and minor triad templates, collapses runs, and weights edges by transition counts. Average size here is eight point three nodes. Neither graph is a musicological parse. Both give Task 2 a well-defined G. We save twenty-four example tensors for the lab.

---

### 4:50–5:40 · Slide 7 · Tasks 1–2

**TinyBERT from scratch, GraphSAGE vs a mel CNN**

TinyBERT is a two-layer pre-norm Transformer, hidden sixty-four, four heads. We train from scratch because five hundred eleven clips will not pretrain bert-base-uncased. The interface still matches HuggingFace CLS, so a real checkpoint can swap in later. GraphSAGE is two dense layers with mean pooling. GAT is implemented with two heads and an adjacency mask. The control is a three-block CNN on sixty-four by thirty-two log-mel. Same tag and genre heads. If the graph is doing extra work, GraphSAGE should beat the CNN. On eight-way genre, the CNN is slightly ahead: 0.489 versus 0.422.

---

### 5:40–6:30 · Slide 8 · Tasks 3–4

**Fuse g with the caption, then align two towers**

Fusion treats the graph vector as a query against caption keys and values, then an MLP on the concatenation. Early concat just stacks g with CLS. Three heads: tags, valence, arousal. Emotion loss weights alpha and beta are 0.35. BERT is warm-started from Task 1. Concat is the workhorse; cross-attention is the harder variant in the brief. Task 4 projects both towers onto a sixty-four-d sphere and trains symmetric InfoNCE at temperature 0.07. We score recall at one, five, and ten on the ninety pairs. Chance R at 5 is five over ninety, about 0.056.

---

### 6:30–7:25 · Slide 9 · Held-out · N = 90

**Text towers dominate tags; audio-only F1 is the honest number**

Random tags: zero macro-F1. Majority: 0.118. CNN: 0.255. GNN: 0.264. BERT-only: 0.772. Concat fusion: 0.783, the best tagger, with AUC-PR 0.868. Cross-attention: 0.779, with a slightly better emotion MAE, 0.037 versus 0.039. Genre: CNN 0.489, GNN 0.422. Caption to audio R at 5 is 0.078, barely above chance 0.056. R at 1 is zero. Rank-1 is often a same-template neighbor, not the paired track. Look at the gap between the text tower and the audio tower. That gap is the result.

---

### 7:25–8:20 · Slide 10 · Reading the table

**BERT is decoding a template we wrote**

Why is BERT so strong? Because the twenty-four tags were extracted from the same metadata that forms the caption. Task 1 is closer to decoding a template than to open-vocabulary MusicCaps tagging. Fusion copies those tags almost exactly. That is expected, not impressive. Audio-only models at about 0.26 F1 are the honest FMA number at this capacity. Emotion MAE looks tiny because the targets are genre priors that both the caption and the graph can recover. It is not a DEAM human-rating result. Retrieval stays near chance because captions share the same bag of tokens. Two substitutions would change the table without touching the code: independent MusicCaps captions, and DEAM listener ratings. We report the FMA run as-is instead of simulating them.

---

### 8:20–9:05 · Slide 11 · Error analysis

**Genre collapse: electronic and hip-hop fall into folk**

Three held-out clips from the lab. Arabesque, folk, Ed Askew: fusion tags match, genre folk to folk. Heartbreaker, experimental electronic: the GNN predicts folk. A hip-hop edit by Laws: the GNN again predicts folk. Folk is the largest mapped class, one hundred twenty-eight clips, and short acoustic chroma is a convenient attractor. Thirty-second graphs at width sixty-four do not solve FMA-small genre. Retrieval also dies when two captions differ only in a title token that TinyBERT treats as unknown after the forty-eight-token cut.

---

### 9:05–10:00 · Slide 12 · Takeaways

**Four tasks on real FMA audio, with the caveats attached**

We implemented the full four-task pipeline on official FMA-small audio, with majority, CNN, BERT, and GNN baselines, concat versus cross-attention, retrieval, a seven-page IEEE report, twenty-four saved graphs, and the lab you can open after this talk. Concat fusion is the best tagger on metadata-derived labels. Audio-only models show the real difficulty. Retrieval needs unique captions. The remaining upgrade is MusicCaps audio and DEAM ratings, not a bigger Transformer on the same leaked tags. Live inference in the lab synthesizes an eight-second clip so the forward pass is visible without redistributing FMA mp3s. Open /slides for this deck, the Paper tab for the IEEE PDF, and Graphs for the twenty-four tensors. I am happy to take questions.

---

**If you are over time:** cut slide 11 after the genre-collapse sentence and go to takeaways. **If you are under:** stay on slide 10 and name the two substitutions (independent MusicCaps captions; DEAM ratings).

Numbers match `results/metrics.json` and `report/final_report.pdf`.
