"use client";

import { useState } from "react";
import { GraphView } from "@/components/GraphView";
import { MelHeatmap } from "@/components/MelHeatmap";
import { TagPills } from "@/components/TagPills";
import { VAPlane } from "@/components/VAPlane";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { MusicGraphJSON, TagScore } from "@/lib/types";

type LiveOut = {
  pred_genre: string;
  genre_true_proxy: string;
  caption: string;
  chord_seq: string[];
  pred_tags: string[];
  pred_tag_scores: TagScore[];
  bert_only_tags: string[];
  pred_valence: number;
  pred_arousal: number;
  proxy_valence: number;
  proxy_arousal: number;
  segment_graph: MusicGraphJSON;
  chord_graph: MusicGraphJSON;
  mel: number[][];
};

const CAPTIONS: Record<string, string> = {
  jazz: "A smoky late-night jazz combo with walking bass and lyrical piano over ii–V–I changes.",
  rock: "Driving rock with overdriven guitars, punchy drums, and a chorus that lands on the tonic.",
  classical: "Chamber strings and a tender piano figure outlining a slow I–vi–IV–V cadence.",
  electronic: "Four-on-the-floor electronics with analog synth stabs and a sidechained bass drop.",
  "hip-hop": "Boom-bap drums, heavy sub bass, and a dry vocal over a minor-key loop.",
  pop: "Radio pop with stacked vocals and a bright I–V–vi–IV chorus hook.",
  metal: "Palm-muted guitars, double-kick drums, and a dark minor riff.",
  folk: "Fingerpicked acoustic guitar, close vocals, and an unhurried story-song cadence.",
};

export function LivePredict({ genres }: { genres: string[] }) {
  const [genre, setGenre] = useState(genres[0] ?? "jazz");
  const [caption, setCaption] = useState(CAPTIONS[genre] ?? CAPTIONS.jazz);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [out, setOut] = useState<LiveOut | null>(null);

  async function run() {
    setBusy(true);
    setErr(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ caption, genre, seed: Date.now() % 10_000 }),
      });
      const text = await res.text();
      let payload: LiveOut & { error?: string; detail?: string };
      try {
        payload = JSON.parse(text) as LiveOut & { error?: string; detail?: string };
      } catch {
        throw new Error(text.slice(0, 200) || `HTTP ${res.status}`);
      }
      if (!res.ok) {
        throw new Error(payload.error || payload.detail || `Inference failed (${res.status})`);
      }
      setOut(payload);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Inference failed. Is the Python API running?");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-1.5">
        {genres.map((g) => (
          <Button
            key={g}
            size="sm"
            variant={genre === g ? "default" : "outline"}
            onClick={() => {
              setGenre(g);
              setCaption(CAPTIONS[g] ?? caption);
            }}
          >
            {g}
          </Button>
        ))}
      </div>
      <textarea
        value={caption}
        onChange={(e) => setCaption(e.target.value)}
        rows={3}
        className="w-full rounded-xl border border-line bg-ink px-3 py-2 text-sm text-cream outline-none focus:border-copper"
      />
      <Button onClick={run} disabled={busy || caption.length < 8}>
        {busy ? "Synthesizing + inferring…" : "Run GNN–BERT fusion"}
      </Button>
      {err ? <p className="text-sm text-amber-200">{err}</p> : null}
      {out ? (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Live clip</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <MelHeatmap data={out.mel} />
              <p className="text-xs text-mute">
                Proxy genre {out.genre_true_proxy} · predicted {out.pred_genre} · {out.chord_seq.join(" → ")}
              </p>
              <div className="h-56 rounded-xl border border-line bg-ink">
                <GraphView graph={out.segment_graph} height={220} />
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Predictions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-[11px] uppercase tracking-widest text-mute">Fusion tags</p>
              <TagPills tags={out.pred_tags} scores={out.pred_tag_scores} />
              <p className="text-[11px] uppercase tracking-widest text-mute">BERT-only tags</p>
              <TagPills tags={out.bert_only_tags} />
              <VAPlane
                valence={out.proxy_valence}
                arousal={out.proxy_arousal}
                predValence={out.pred_valence}
                predArousal={out.pred_arousal}
              />
            </CardContent>
          </Card>
        </div>
      ) : null}
    </div>
  );
}
