"use client";

import { GraphsView } from "@/components/GraphsView";
import { PaperView } from "@/components/PaperView";
import { Explorer } from "@/components/Explorer";
import { LivePredict } from "@/components/LivePredict";
import { AblationChart, F1Curves, MetricsTable } from "@/components/Metrics";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { DemoPayload } from "@/lib/types";
import { fmt } from "@/lib/utils";

const NAV = [
  ["#lab", "Overview"],
  ["#explorer", "Explorer"],
  ["#graphs", "Graphs"],
  ["#live", "Live inference"],
  ["#metrics", "Metrics"],
  ["#retrieval", "Retrieval"],
  ["#cases", "Case studies"],
  ["#paper", "Paper"],
  ["/slides", "Slides"],
];

export function LabApp({ demo }: { demo: DemoPayload | null }) {

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-line/80 bg-ink/85 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3">
          <a href="#lab" className="shrink-0 font-serif text-lg tracking-tight">
            Cadence Lab
          </a>
          <nav className="hidden gap-4 text-xs uppercase tracking-widest text-mute md:flex">
            {NAV.map(([href, label]) => (
              <a key={href} href={href} className="shrink-0 hover:text-cream">
                {label}
              </a>
            ))}
          </nav>
          <a
            href="/gnn-bert-music-context.zip"
            download
            className="shrink-0 rounded-full bg-copper px-3 py-1.5 text-xs font-medium text-ink hover:bg-copper/90"
          >
            Download zip
          </a>
        </div>
        <nav className="mx-auto flex max-w-6xl gap-3 overflow-x-auto px-4 pb-2 text-[11px] uppercase tracking-widest text-mute md:hidden">
          {NAV.map(([href, label]) => (
            <a key={href} href={href} className="shrink-0 hover:text-cream">
              {label}
            </a>
          ))}
        </nav>
      </header>

      <main className="mx-auto max-w-6xl px-4 pb-24">
        <section id="lab" className="grid gap-10 py-14 lg:grid-cols-12 lg:py-20">
          <div className="lg:col-span-7">
            <p className="text-[11px] uppercase tracking-[0.25em] text-copper">CSE425 · Neural Networks</p>
            <h1 className="mt-3 font-serif text-4xl leading-tight text-cream md:text-6xl">
              GNN-based BERT for musical context
            </h1>
            <p className="mt-5 max-w-xl text-lg text-mute">
              A hybrid GraphSAGE + TinyBERT system trained on a 511-clip FMA-small subset
              {demo?.corpus ? ` (${demo.corpus.n_tracks} ${demo.corpus.source} tracks)` : ""}. It reads a
              track as a structure graph and a caption, then predicts multi-label tags, valence/arousal,
              and retrieves matching clips.
            </p>
            <div className="mt-6 flex flex-wrap gap-2">
              <Badge tone="copper">Task 1 BERT tags</Badge>
              <Badge tone="teal">Task 2 GraphSAGE</Badge>
              <Badge tone="copper">Task 3 cross-attention</Badge>
              <Badge tone="teal">Task 4 InfoNCE</Badge>
            </div>
          </div>
          <div className="lg:col-span-5 space-y-3">
            {[
              ["Audio graph", "1 s segments, temporal + chroma/MFCC similarity edges, chord-transition graph."],
              ["Text encoder", "CLS contextualization of MusicCaps-style captions (swap-in for bert-base-uncased)."],
              ["Fusion", "g attends over H_text; multi-task BCE tags + MSE valence/arousal."],
            ].map(([t, d]) => (
              <div key={t} className="rounded-2xl border border-line bg-panel px-4 py-3">
                <p className="text-sm text-cream">{t}</p>
                <p className="text-sm text-mute">{d}</p>
              </div>
            ))}
          </div>
        </section>

        {demo ? null : (
          <Card>
            <CardHeader>
              <CardTitle>Waiting on trained artifacts</CardTitle>
              <CardDescription>public/data/demo.json is missing. Train, then evaluate.</CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="overflow-auto rounded-xl bg-ink p-4 text-xs text-mute">
{`PYTHONPATH=. python scripts/build_corpus.py
PYTHONPATH=. python -m src.train --task all
PYTHONPATH=. python -m src.evaluate`}
              </pre>
            </CardContent>
          </Card>
        )}

        {demo ? (
          <>
            <section className="mb-16 grid gap-3 sm:grid-cols-3">
              {demo.project.tasks.map((t) => (
                <div key={t.id} className="rounded-2xl border border-line bg-panel px-4 py-4">
                  <p className="text-[11px] uppercase tracking-widest text-copper">Task {t.id} · {t.marks} marks</p>
                  <p className="mt-1 font-serif text-xl">{t.name}</p>
                </div>
              ))}
            </section>

            <section id="explorer" className="mb-20">
              <h2 className="font-serif text-3xl">Track explorer</h2>
              <p className="mb-6 mt-2 max-w-2xl text-mute">
                Each clip is a paired (graph, caption, tags, valence/arousal) example. Open a track to
                inspect the log-mel, the structure graph, and the fusion model’s predictions.
              </p>
              <Explorer demo={demo} />
            </section>

            <section id="graphs" className="mb-20">
              <GraphsView demo={demo} />
            </section>

            <section id="live" className="mb-20">
              <h2 className="font-serif text-3xl">Live inference</h2>
              <p className="mb-6 mt-2 max-w-2xl text-mute">
                Synthesize a new 8 s clip from a genre recipe, build its graphs on the fly, and run the
                trained GNN–BERT fusion head.
              </p>
              <LivePredict genres={demo.taxonomy.genres} />
            </section>

            <section id="metrics" className="mb-20">
              <h2 className="font-serif text-3xl">Ablations &amp; curves</h2>
              <p className="mb-6 mt-2 max-w-2xl text-mute">
                Held-out artist split ({demo.splits.train} / {demo.splits.val} / {demo.splits.test} tracks).
                Cross-attention fusion is the Task 3 headline model.
              </p>
              <div className="grid gap-5 lg:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle>Macro-F1 vs epoch</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <F1Curves demo={demo} />
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Test ablation</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <AblationChart demo={demo} />
                  </CardContent>
                </Card>
              </div>
              <Card className="mt-5">
                <CardHeader>
                  <CardTitle>Evaluation table</CardTitle>
                  <CardDescription>
                    Held-out FMA-small subset (artist-disjoint). MusicCaps captions CSV is on disk; DEAM audio was not available so valence/arousal use genre priors.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <MetricsTable demo={demo} />
                </CardContent>
              </Card>
              <div className="mt-5 grid gap-4 md:grid-cols-2">
                <figure className="rounded-2xl border border-line bg-panel p-3">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src="/plots/f1_curves.png" alt="Macro-F1 training curves" className="w-full rounded-xl bg-white" />
                  <figcaption className="mt-2 text-xs text-mute">Macro-F1 vs epoch (same curves as the chart)</figcaption>
                </figure>
                <figure className="rounded-2xl border border-line bg-panel p-3">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src="/plots/ablation.png" alt="Test ablation bar chart" className="w-full rounded-xl bg-white" />
                  <figcaption className="mt-2 text-xs text-mute">Held-out ablation (macro-F1 and AUC-PR)</figcaption>
                </figure>
                <figure className="rounded-2xl border border-line bg-panel p-3">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src="/plots/tsne_genre.png" alt="t-SNE of fusion vectors coloured by genre" className="w-full rounded-xl" />
                  <figcaption className="mt-2 text-xs text-mute">t-SNE of fusion vector z, coloured by genre</figcaption>
                </figure>
                <figure className="rounded-2xl border border-line bg-panel p-3">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src="/plots/valence_arousal.png" alt="Valence-arousal plane" className="w-full rounded-xl" />
                  <figcaption className="mt-2 text-xs text-mute">DEAM-style valence / arousal plane</figcaption>
                </figure>
              </div>
            </section>

            <section id="retrieval" className="mb-20">
              <h2 className="font-serif text-3xl">Caption → audio retrieval</h2>
              <p className="mb-6 mt-2 max-w-2xl text-mute">
                Dual-encoder InfoNCE. A query caption should rank its paired graph in the top-k.
                R@1 / R@5 / R@10 = {fmt(demo.metrics.models.contrastive?.["caption_to_audio_R@1"] ?? NaN)} /{" "}
                {fmt(demo.metrics.models.contrastive?.["caption_to_audio_R@5"] ?? NaN)} /{" "}
                {fmt(demo.metrics.models.contrastive?.["caption_to_audio_R@10"] ?? NaN)}.
              </p>
              <div className="space-y-3">
                {demo.retrieval.map((row) => (
                  <Card key={row.query_id}>
                    <CardHeader>
                      <CardTitle className="text-lg">{row.query_title}</CardTitle>
                      <CardDescription>
                        {row.query_genre} · {row.query_caption}
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ol className="space-y-1.5 text-sm">
                        {row.hits.map((h, i) => (
                          <li key={h.id} className="flex items-start justify-between gap-3">
                            <span>
                              <span className="text-mute">{i + 1}.</span> {h.title}{" "}
                              <span className="text-mute">({h.genre})</span>
                            </span>
                            <span className={h.correct ? "text-teal" : "text-mute"}>
                              {fmt(h.score, 2)} {h.correct ? "match" : ""}
                            </span>
                          </li>
                        ))}
                      </ol>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </section>

            <section id="cases" className="mb-10">
              <h2 className="font-serif text-3xl">Case studies</h2>
              <p className="mb-6 mt-2 max-w-2xl text-mute">
                Three held-out clips. Follow the chord path, the caption, and whether fusion recovers
                the tags a listener would name.
              </p>
              <div className="grid gap-4 md:grid-cols-3">
                {demo.case_ids.map((id) => {
                  const t = demo.tracks.find((x) => x.id === id);
                  if (!t) return null;
                  return (
                    <Card key={id}>
                      <CardHeader>
                        <CardTitle>{t.title}</CardTitle>
                        <CardDescription>
                          {t.genre} → pred {t.pred_genre}
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-3 text-sm text-mute">
                        <p className="text-cream/90">{t.caption}</p>
                        <p className="font-mono text-xs">{t.chord_seq.join(" → ")}</p>
                        <p>
                          tags true: {t.tags.join(", ")}
                          <br />
                          tags pred: {t.pred_tags.join(", ")}
                        </p>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </section>
            <section id="paper" className="mb-10">
              <PaperView />
            </section>
            <section id="slides" className="mb-10">
              <div className="flex flex-wrap items-end justify-between gap-3">
                <div>
                  <h2 className="font-serif text-3xl">Course slides</h2>
                  <p className="mt-2 max-w-2xl text-mute">
                    Twelve-slide talk with an 8–10 minute speaker script. Open the web deck,
                    download the PowerPoint, or read the timed script. On the web deck, press{" "}
                    <span className="text-cream">N</span> for notes.
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <a
                    href="/cadence-lab-slides.pptx"
                    download
                    className="rounded-full bg-copper px-4 py-2 text-sm font-medium text-ink hover:bg-copper/90"
                  >
                    Download PowerPoint
                  </a>
                  <a
                    href="/slides"
                    className="rounded-full border border-line px-4 py-2 text-sm text-cream hover:bg-white/5"
                  >
                    Open web slides
                  </a>
                  <a
                    href="/slides/script"
                    className="rounded-full border border-line px-4 py-2 text-sm text-cream hover:bg-white/5"
                  >
                    Speaker script
                  </a>
                </div>
              </div>
            </section>
          </>
        ) : null}
      </main>
      <footer className="border-t border-line py-8 text-center text-xs text-mute">
        Cadence Lab · course project for CSE425 / EEE474 / CSE715 · not a generative music model
      </footer>
    </div>
  );
}
