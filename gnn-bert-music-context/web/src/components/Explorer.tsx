"use client";

import { useMemo, useState } from "react";
import { GraphView } from "@/components/GraphView";
import { MelHeatmap } from "@/components/MelHeatmap";
import { TagPills } from "@/components/TagPills";
import { VAPlane } from "@/components/VAPlane";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { DemoPayload, Track } from "@/lib/types";
import { fmt } from "@/lib/utils";

export function Explorer({ demo }: { demo: DemoPayload }) {
  const [genre, setGenre] = useState<string>("all");
  const [graphKind, setGraphKind] = useState<"segment" | "chord">("segment");
  const filtered = useMemo(
    () => demo.tracks.filter((t) => (genre === "all" ? true : t.genre === genre)),
    [demo.tracks, genre],
  );
  const [idx, setIdx] = useState(0);
  const track: Track | undefined = filtered[Math.min(idx, Math.max(0, filtered.length - 1))];
  if (!track) {
    return <p className="text-mute">No tracks in this filter.</p>;
  }
  const graph = graphKind === "chord" ? track.chord_graph : track.segment_graph;
  return (
    <div className="grid gap-5 lg:grid-cols-12">
      <div className="lg:col-span-4 space-y-3">
        <div className="flex flex-wrap gap-1.5">
          <Button size="sm" variant={genre === "all" ? "default" : "outline"} onClick={() => { setGenre("all"); setIdx(0); }}>
            all
          </Button>
          {demo.taxonomy.genres.map((g) => (
            <Button key={g} size="sm" variant={genre === g ? "default" : "outline"} onClick={() => { setGenre(g); setIdx(0); }}>
              {g}
            </Button>
          ))}
        </div>
        <div className="max-h-[540px] space-y-2 overflow-auto pr-1">
          {filtered.map((t, i) => (
            <button
              key={t.id}
              onClick={() => setIdx(i)}
              className={`w-full rounded-xl border px-3 py-2.5 text-left transition ${
                i === idx ? "border-copper bg-copper/10" : "border-line hover:border-copper/40"
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium text-cream">{t.title}</span>
                <Badge tone="teal">{t.genre}</Badge>
              </div>
              <p className="mt-1 line-clamp-2 text-xs text-mute">{t.caption}</p>
            </button>
          ))}
        </div>
      </div>
      <div className="lg:col-span-8 space-y-4">
        <Card>
          <CardHeader className="flex flex-row items-start justify-between gap-3">
            <div>
              <CardTitle>{track.title}</CardTitle>
              <p className="mt-1 text-sm text-mute">{track.caption}</p>
            </div>
            <div className="flex gap-2">
              <Badge tone={track.pred_genre === track.genre ? "ok" : "warn"}>
                pred {track.pred_genre}
              </Badge>
              <Badge>{track.split}</Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <MelHeatmap data={track.mel} />
            <div className="flex flex-wrap gap-2">
              <Button size="sm" variant={graphKind === "segment" ? "default" : "outline"} onClick={() => setGraphKind("segment")}>
                Segment graph
              </Button>
              <Button size="sm" variant={graphKind === "chord" ? "default" : "outline"} onClick={() => setGraphKind("chord")}>
                Chord graph
              </Button>
              <span className="self-center text-xs text-mute">
                {graph.nodes.length} nodes · {graph.edges.length} edges · coherence {fmt(track.graph_coherence)}
              </span>
            </div>
            <div className="h-[300px] rounded-xl border border-line bg-ink">
              <GraphView graph={graph} height={300} />
            </div>
            {track.chord_seq?.length ? (
              <p className="text-xs text-mute">
                Chord path: <span className="font-mono text-cream/80">{track.chord_seq.join(" → ")}</span>
              </p>
            ) : null}
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <p className="mb-2 text-[11px] uppercase tracking-widest text-mute">True tags</p>
                <TagPills tags={track.tags} />
                <p className="mb-2 mt-4 text-[11px] uppercase tracking-widest text-mute">Fusion predictions</p>
                <TagPills tags={track.pred_tags} scores={track.pred_tag_scores} truth={track.tags} />
              </div>
              <div>
                <p className="mb-2 text-[11px] uppercase tracking-widest text-mute">Valence / arousal</p>
                <VAPlane
                  valence={track.valence}
                  arousal={track.arousal}
                  predValence={track.pred_valence}
                  predArousal={track.pred_arousal}
                />
                <p className="mt-2 text-xs text-mute">
                  true ({fmt(track.valence)}, {fmt(track.arousal)}) · pred ({fmt(track.pred_valence)}, {fmt(track.pred_arousal)})
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
