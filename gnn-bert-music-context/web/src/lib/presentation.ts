/** Course talk: 12 slides, ~9 minutes at a measured speaking pace. */

export type Metric = { label: string; value: string; hint?: string };

export type Slide = {
  id: string;
  kicker: string;
  title: string;
  durationSec: number;
  script: string;
  bullets?: string[];
  metrics?: Metric[];
  footnote?: string;
  layout: "title" | "bullets" | "metrics" | "split" | "close";
  split?: { heading: string; items: string[] }[];
};

export const TALK_TITLE = "GNN-Based BERT for Understanding Context from Music";
export const TALK_COURSE = "CSE425 / EEE474 / CSE715 — Neural Networks";
export const TALK_TARGET = "8–10 minutes";

export const SLIDES: Slide[] = [
  {
    id: "title",
    kicker: "Course presentation",
    title: TALK_TITLE,
    layout: "title",
    durationSec: 45,
    script:
      "This is Cadence Lab’s course project for Neural Networks: GNN-based BERT for understanding context from music. The assignment is not to generate audio. It is to read a clip as structure plus language, then predict tags, genre, mood, and retrieval. Everything I will show was trained on official Free Music Archive small audio: five hundred eleven thirty-second clips, artist-disjoint splits. I will be honest about where the labels leak from metadata, because that is the actual finding, not a leaderboard claim. After the numbers I will point you to the lab, the saved graphs, and the seven-page IEEE report.",
    bullets: [
      "Understanding, not generation",
      "Official FMA-small subset · 511 clips · 386 artists",
      "Four tasks: BERT tags, GraphSAGE, fusion, InfoNCE",
    ],
    footnote: "Cadence Lab · September 2026",
  },
  {
    id: "why",
    kicker: "Motivation",
    title: "A clip is harmonic, timbral, and linguistic at once",
    layout: "split",
    durationSec: 50,
    script:
      "A recording can be folk, acoustic, guitar-led, and built from a short chord loop at the same time. A spectrogram CNN hears local timbre. It does not see that bar three repeats bar one, or that A minor goes to C major. A language model on a caption hears “a folk recording titled Arabesque by Ed Askew.” It does not know when those words line up with a modulation. The brief therefore asks for a hybrid: BERT on text, a graph net on a structure graph from audio, a fusion stage, and a contrastive retriever. Five beats in this talk: why, representation, the four models, held-out numbers, then what those numbers actually mean.",
    split: [
      {
        heading: "CNN on log-mel",
        items: ["Local timbre and transients", "Blind to chord loops", "Blind to repeated segments"],
      },
      {
        heading: "LM on captions",
        items: ["Instruments, genre, mood words", "No alignment to time", "Cannot hear a drop"],
      },
      {
        heading: "This project",
        items: ["Graph G from the spectrogram", "TinyBERT on the caption", "Fuse, then retrieve"],
      },
    ],
  },
  {
    id: "tasks",
    kicker: "Assignment",
    title: "Four tasks, one pipeline",
    layout: "split",
    durationSec: 45,
    script:
      "Task one is easy on paper: a BERT tag classifier on captions, twenty-four MagnaTagATune-style labels, binary cross-entropy. Task two is the graph: GraphSAGE, and GAT, on a segment-similarity graph and a chord-transition graph, against a CNN mel baseline. Task three fuses the graph readout with token states, concat versus cross-attention, plus valence and arousal heads. Task four is a dual encoder with InfoNCE. We also ship majority and random tag baselines so the table is complete.",
    split: [
      { heading: "1  BERT tags", items: ["TinyBERT CLS", "24-way BCE", "Swap-in for bert-base"] },
      { heading: "2  Graphs", items: ["GraphSAGE / GAT", "Segment + chord G", "CNN control"] },
      { heading: "3  Fusion", items: ["Concat vs cross-attn", "Tag + VA heads", "Warm-start BERT"] },
      { heading: "4  Retrieval", items: ["Two towers", "InfoNCE τ = 0.07", "R@1 / 5 / 10"] },
    ],
  },
  {
    id: "tuple",
    kicker: "Representation",
    title: "A track is the tuple T = (audio, text, G, y)",
    layout: "bullets",
    durationSec: 40,
    script:
      "Audio is a one-twenty-eight-bin log-mel at twenty-two thousand fifty Hertz, hop five twelve, plus chroma, MFCCs, RMS, centroid, and zero-crossing rate, all z-scored per clip. Text is a whitespace-tokenized caption of at most forty-eight tokens with CLS and SEP. G is either a thirty-node segment graph or a chord-transition graph. Labels y are twenty-four binary tags, eight-way genre, and valence-arousal in zero to one. Jazz and metal heads exist in the classifier but FMA-small has no such top genres, so those rows stay empty.",
    bullets: [
      "X_audio: log-mel 128, chroma 12, MFCC 13, RMS, centroid, ZCR",
      "X_text: ≤ 48 tokens, vocabulary rebuilt from FMA captions",
      "G: 30-node segment graph or ~8-node chord graph",
      "y: 24 tags · 8 genres · valence / arousal in [0, 1]",
    ],
  },
  {
    id: "data",
    kicker: "Corpus",
    title: "Official FMA-small, not a synthetic stand-in",
    layout: "metrics",
    durationSec: 55,
    script:
      "We fetched FMA metadata and FMA-small from the official SWITCH mirror and SHA1-checked both zips. Seed forty-two, sixty-four clips from each of eight top genres. One mp3 failed to decode, leaving five hundred eleven tracks from three hundred eighty-six artists. Splits are artist-disjoint: three hundred twenty-seven, ninety-four, ninety. After mapping, folk and electronic have one hundred twenty-eight clips each; hip-hop has sixty-three. Experimental maps to electronic, instrumental to classical, international to folk. Captions are templates from title, artist, album, and leaf genres. MusicCaps CSV is on disk without YouTube audio. DEAM is gated, so valence and arousal are genre priors. Tags are keyword-mapped from those captions. That last fact will explain BERT’s F1.",
    metrics: [
      { label: "Clips", value: "511", hint: "30 s · 22 050 Hz" },
      { label: "Artists", value: "386", hint: "no artist in two splits" },
      { label: "Test", value: "90", hint: "held-out tracks" },
      { label: "Tags / clip", value: "6.24", hint: "metadata-mapped" },
    ],
    footnote: "MusicCaps audio and DEAM ratings are not on disk. Jazz / metal top-genres are unused.",
  },
  {
    id: "graphs",
    kicker: "Task 2",
    title: "Two graphs, both deterministic functions of the spectrogram",
    layout: "split",
    durationSec: 55,
    script:
      "The segment graph has thirty windows of one second. Each node concatenates mean and standard deviation of the audio streams: three hundred twelve dimensions. Adjacent windows get a temporal edge. If chroma concatenated with MFCC cosine exceeds tau 0.62, we add a similarity edge. That is a self-similarity matrix with a backbone. The chord graph matches frames to twenty-four major and minor triad templates, collapses runs, and weights edges by transition counts. Average size here is eight point three nodes. Neither graph is a musicological parse. Both give Task 2 a well-defined G. We save twenty-four example tensors for the lab.",
    split: [
      {
        heading: "Segment graph",
        items: ["N = 30 · 1 s windows", "Node dim 312", "Temporal + cosine > 0.62", "Self-loops kept"],
      },
      {
        heading: "Chord graph",
        items: ["24 maj / min templates", "Collapse consecutive ids", "Edges = transition counts", "Mean N ≈ 8.3"],
      },
    ],
    footnote: "Dense batched GraphSAGE / GAT. No PyTorch Geometric.",
  },
  {
    id: "encoders",
    kicker: "Tasks 1–2",
    title: "TinyBERT from scratch, GraphSAGE vs a mel CNN",
    layout: "bullets",
    durationSec: 50,
    script:
      "TinyBERT is a two-layer pre-norm Transformer, hidden sixty-four, four heads. We train from scratch because five hundred eleven clips will not pretrain bert-base-uncased. The interface still matches HuggingFace CLS, so a real checkpoint can swap in later. GraphSAGE is two dense layers with mean pooling. GAT is implemented with two heads and an adjacency mask. The control is a three-block CNN on sixty-four by thirty-two log-mel. Same tag and genre heads. If the graph is doing extra work, GraphSAGE should beat the CNN. On eight-way genre, the CNN is slightly ahead: 0.489 versus 0.422.",
    bullets: [
      "TinyBERT: 2 layers, d = 64, 4 heads, dropout 0.15, L = 48",
      "GraphSAGE: 2 layers, mean pool; GAT optional (2 heads, leaky 0.2)",
      "CNN: 16–32–64 channels on 64 × 32 log-mel",
      "AdamW 1.5×10⁻³, batch 16, CPU, seed 42",
    ],
  },
  {
    id: "fusion",
    kicker: "Tasks 3–4",
    title: "Fuse g with the caption, then align two towers",
    layout: "split",
    durationSec: 50,
    script:
      "Fusion treats the graph vector as a query against caption keys and values, then an MLP on the concatenation. Early concat just stacks g with CLS. Three heads: tags, valence, arousal. Emotion loss weights alpha and beta are 0.35. BERT is warm-started from Task 1. Concat is the workhorse; cross-attention is the harder variant in the brief. Task 4 projects both towers onto a sixty-four-d sphere and trains symmetric InfoNCE at temperature 0.07. We score recall at one, five, and ten on the ninety pairs. Chance R at 5 is five over ninety, about 0.056.",
    split: [
      {
        heading: "Fusion",
        items: ["Q = g W_Q, K,V from tokens", "Concat ablation: [g ; CLS]", "L = BCE + 0.35 VA L2"],
      },
      {
        heading: "InfoNCE",
        items: ["Two 64-d spheres", "Symmetric batch loss", "R@K both directions"],
      },
    ],
  },
  {
    id: "results",
    kicker: "Held-out · N = 90",
    title: "Text towers dominate tags; audio-only F1 is the honest number",
    layout: "metrics",
    durationSec: 55,
    script:
      "Random tags: zero macro-F1. Majority: 0.118. CNN: 0.255. GNN: 0.264. BERT-only: 0.772. Concat fusion: 0.783, the best tagger, with AUC-PR 0.868. Cross-attention: 0.779, with a slightly better emotion MAE, 0.037 versus 0.039. Genre: CNN 0.489, GNN 0.422. Caption to audio R at 5 is 0.078, barely above chance 0.056. R at 1 is zero. Rank-1 is often a same-template neighbor, not the paired track. Look at the gap between the text tower and the audio tower. That gap is the result.",
    metrics: [
      { label: "Concat F1", value: "0.783", hint: "best tagger" },
      { label: "BERT F1", value: "0.772", hint: "caption only" },
      { label: "GNN / CNN F1", value: "0.26", hint: "audio only" },
      { label: "C→A R@5", value: "0.078", hint: "chance 0.056" },
    ],
    footnote: "AUC-PR concat 0.868 · genre CNN 0.489 vs GNN 0.422 · VA MAE 0.037 (priors, not DEAM).",
  },
  {
    id: "honest",
    kicker: "Reading the table",
    title: "BERT is decoding a template we wrote",
    layout: "bullets",
    durationSec: 55,
    script:
      "Why is BERT so strong? Because the twenty-four tags were extracted from the same metadata that forms the caption. Task 1 is closer to decoding a template than to open-vocabulary MusicCaps tagging. Fusion copies those tags almost exactly. That is expected, not impressive. Audio-only models at about 0.26 F1 are the honest FMA number at this capacity. Emotion MAE looks tiny because the targets are genre priors that both the caption and the graph can recover. It is not a DEAM human-rating result. Retrieval stays near chance because captions share the same bag of tokens. Two substitutions would change the table without touching the code: independent MusicCaps captions, and DEAM listener ratings. We report the FMA run as-is instead of simulating them.",
    bullets: [
      "Template leakage: tags ⊂ caption metadata → inflated BERT / fusion F1",
      "Audio-only ~0.26 F1 is the real FMA tagging difficulty here",
      "VA MAE 0.037 recovers genre priors, not listener ratings",
      "FMA captions are stereotyped → InfoNCE cannot identify the pair",
    ],
  },
  {
    id: "cases",
    kicker: "Error analysis",
    title: "Genre collapse: electronic and hip-hop fall into folk",
    layout: "split",
    durationSec: 45,
    script:
      "Three held-out clips from the lab. Arabesque, folk, Ed Askew: fusion tags match, genre folk to folk. Heartbreaker, experimental electronic: the GNN predicts folk. A hip-hop edit by Laws: the GNN again predicts folk. Folk is the largest mapped class, one hundred twenty-eight clips, and short acoustic chroma is a convenient attractor. Thirty-second graphs at width sixty-four do not solve FMA-small genre. Retrieval also dies when two captions differ only in a title token that TinyBERT treats as unknown after the forty-eight-token cut.",
    split: [
      { heading: "000620 Arabesque", items: ["Caption: folk, Ed Askew", "Tags: copied", "Genre: folk → folk"] },
      { heading: "007373 heartbreaker", items: ["Caption: electronic", "Tags: drops dark", "Genre: elec. → folk"] },
      { heading: "013749 110% edit", items: ["Caption: hip-hop, Laws", "Tags: copied", "Genre: hip-hop → folk"] },
    ],
  },
  {
    id: "close",
    kicker: "Takeaways",
    title: "Four tasks on real FMA audio, with the caveats attached",
    layout: "close",
    durationSec: 55,
    script:
      "We implemented the full four-task pipeline on official FMA-small audio, with majority, CNN, BERT, and GNN baselines, concat versus cross-attention, retrieval, a seven-page IEEE report, twenty-four saved graphs, and the lab you can open after this talk. Concat fusion is the best tagger on metadata-derived labels. Audio-only models show the real difficulty. Retrieval needs unique captions. The remaining upgrade is MusicCaps audio and DEAM ratings, not a bigger Transformer on the same leaked tags. Live inference in the lab synthesizes an eight-second clip so the forward pass is visible without redistributing FMA mp3s. Open /slides for this deck, the Paper tab for the IEEE PDF, and Graphs for the twenty-four tensors. I am happy to take questions.",
    bullets: [
      "Concat fusion 0.783 F1 — on labels built from captions",
      "GNN 0.264 / CNN 0.255 — honest audio tagging",
      "R@5 0.078 vs chance 0.056 — report the miss",
      "Lab: graphs, metrics, live fusion, IEEE paper",
    ],
    footnote: "Questions  ·  /slides  ·  report/final_report.pdf",
  },
];

export const TOTAL_SEC = SLIDES.reduce((s, d) => s + d.durationSec, 0);

export function scriptWordCount(): number {
  return SLIDES.reduce((n, s) => n + s.script.trim().split(/\s+/).length, 0);
}
