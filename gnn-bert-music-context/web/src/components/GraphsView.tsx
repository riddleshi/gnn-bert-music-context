"use client";

import { useEffect, useMemo, useState } from "react";
import { GraphView } from "@/components/GraphView";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { DemoPayload, MusicGraphJSON } from "@/lib/types";

const SAMPLES = Array.from({ length: 12 }, (_, i) => `track_${String(i).padStart(3, "0")}`);

type Loaded = MusicGraphJSON & {
  file?: string;
  keys?: string[];
  node_labels?: string[];
  x_shape?: number[];
  source?: string;
};

function fromDemo(demo: DemoPayload, trackId: string, kind: "segment" | "chord"): Loaded | null {
  const t = demo.tracks.find((x) => x.id === trackId);
  if (!t) return null;
  const g = kind === "chord" ? t.chord_graph : t.segment_graph;
  if (!g?.nodes?.length) return null;
  return {
    ...g,
    file: `${trackId}_${kind}.pt`,
    node_labels: g.nodes.map((n) => n.label),
    source: "demo",
  };
}

async function loadGraph(ptFile: string, jsonFile: string): Promise<Loaded> {
  try {
    const live = await fetch(`/api/graph?file=${encodeURIComponent(ptFile)}`);
    if (live.ok) {
      const data = (await live.json()) as Loaded;
      if (data.nodes?.length) return data;
    }
  } catch {
    // fall through
  }
  const fallback = await fetch(`/graphs/${jsonFile}`);
  if (!fallback.ok) {
    throw new Error(`Could not load ${ptFile}`);
  }
  const data = (await fallback.json()) as Loaded;
  return { ...data, source: "json-fallback", file: ptFile };
}

export function GraphsView({ demo }: { demo: DemoPayload }) {
  const [trackId, setTrackId] = useState(SAMPLES[0]);
  const [kind, setKind] = useState<"segment" | "chord">("segment");
  const [live, setLive] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pngOk, setPngOk] = useState(true);

  const ptFile = `${trackId}_${kind}.pt`;
  const jsonFile = `${trackId}_${kind}.json`;
  const pngFile = `/graphs/${trackId}_${kind}.png`;
  const meta = demo.tracks.find((t) => t.id === trackId);
  const demoGraph = useMemo(() => fromDemo(demo, trackId, kind), [demo, trackId, kind]);
  const graph = live ?? demoGraph;

  useEffect(() => {
    let cancelled = false;
    setLive(null);
    setError(null);
    setPngOk(true);
    loadGraph(ptFile, jsonFile)
      .then((data) => {
        if (!cancelled) setLive(data);
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [ptFile, jsonFile]);

  return (
    <div className="space-y-5">
      <div>
        <h2 className="font-serif text-3xl">Preprocessed graph samples</h2>
        <p className="mt-2 max-w-2xl text-mute">
          24 saved graphs (12 clips × segment + chord). Each panel is the same{" "}
          <code className="text-cream">{ptFile}</code> tensor, drawn as an SVG and as a matplotlib figure.
        </p>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {SAMPLES.map((id) => (
          <Button
            key={id}
            size="sm"
            variant={id === trackId ? "default" : "outline"}
            onClick={() => setTrackId(id)}
          >
            {id.replace("track_", "")}
          </Button>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant={kind === "segment" ? "default" : "outline"} onClick={() => setKind("segment")}>
          Segment graph
        </Button>
        <Button size="sm" variant={kind === "chord" ? "default" : "outline"} onClick={() => setKind("chord")}>
          Chord graph
        </Button>
        <Badge tone="teal">{ptFile}</Badge>
        {graph ? (
          <span className="text-xs text-mute">
            {graph.nodes.length} nodes · {graph.edges.length} edges
            {meta ? ` · ${meta.title}` : ""}
          </span>
        ) : null}
      </div>

      <div className="overflow-hidden rounded-2xl border border-line bg-[#1c1814]">
        <div className={pngOk ? "grid gap-0 lg:grid-cols-2" : ""}>
          <div className="h-[380px] border-b border-line lg:border-b-0 lg:border-r">
            {graph ? (
              <GraphView graph={graph} height={380} />
            ) : error ? (
              <p className="p-8 text-sm text-amber-200">{error}</p>
            ) : (
              <p className="p-8 text-sm text-mute">Drawing {ptFile}…</p>
            )}
          </div>
          {pngOk ? (
            <div className="flex h-[380px] items-center justify-center bg-[#141210] p-3">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={pngFile}
                alt={`${ptFile} matplotlib drawing`}
                className="max-h-full max-w-full object-contain"
                onError={() => setPngOk(false)}
              />
            </div>
          ) : null}
        </div>
      </div>

      {graph?.nodes?.length ? (
        <p className="text-sm text-cream">
          node labels:{" "}
          <span className="font-mono text-mute">
            {(graph.node_labels ?? graph.nodes.map((n) => n.label)).join(" → ")}
          </span>
        </p>
      ) : null}

      <details className="rounded-2xl border border-line bg-panel px-5 py-4">
        <summary className="cursor-pointer text-sm text-cream">Python used to load this file</summary>
        <pre className="mt-3 overflow-auto text-xs text-mute">{`import torch

graph = torch.load(
    "data/processed/graphs/${ptFile}",
    weights_only=False,
)

print(graph.keys())
print(graph["node_labels"])`}</pre>
      </details>
    </div>
  );
}
