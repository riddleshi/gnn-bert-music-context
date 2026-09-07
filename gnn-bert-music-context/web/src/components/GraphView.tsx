"use client";

import type { MusicGraphJSON } from "@/lib/types";

const KIND_COLOR: Record<string, string> = {
  segment: "#5ee0c0",
  chord: "#e8a05a",
};

function num(v: unknown, fallback = 0): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export function GraphView({ graph, height = 320 }: { graph: MusicGraphJSON; height?: number }) {
  const nodes = graph.nodes ?? [];
  const edges = graph.edges ?? [];
  if (!nodes.length) {
    return (
      <div className="flex h-full min-h-[240px] items-center justify-center text-sm text-mute">
        No graph for this clip.
      </div>
    );
  }
  const xs = nodes.map((n) => num(n.x));
  const ys = nodes.map((n) => num(n.y));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const pad = 56;
  const W = 640;
  const H = Math.max(height, 280);
  const dx = maxX - minX || 1;
  const dy = maxY - minY || 1;
  const sx = (x: number) => pad + ((x - minX) / dx) * (W - pad * 2);
  const sy = (y: number) => pad + ((y - minY) / dy) * (H - pad * 2);
  const pos = new Map(nodes.map((n) => [num(n.id), { x: sx(num(n.x)), y: sy(num(n.y)) }]));
  const stroke = KIND_COLOR[graph.kind] ?? "#e8a05a";

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-full w-full"
      role="img"
      aria-label={`${graph.kind} graph with ${nodes.length} nodes`}
    >
      <rect width={W} height={H} fill="#1c1814" rx="12" />
      {edges.map((e, i) => {
        const a = pos.get(num(e.source));
        const b = pos.get(num(e.target));
        if (!a || !b) return null;
        const w = Math.min(1, num(e.weight, 0.4));
        return (
          <line
            key={i}
            x1={a.x}
            y1={a.y}
            x2={b.x}
            y2={b.y}
            stroke={stroke}
            strokeOpacity={0.35 + 0.55 * w}
            strokeWidth={2 + 3 * w}
          />
        );
      })}
      {nodes.map((n) => {
        const p = pos.get(num(n.id));
        if (!p) return null;
        return (
          <g key={String(n.id)}>
            <circle cx={p.x} cy={p.y} r={14} fill={stroke} stroke="#efe6d6" strokeWidth={1.6} />
            <text
              x={p.x}
              y={p.y + 28}
              textAnchor="middle"
              fill="#efe6d6"
              fontSize={11}
              fontFamily="ui-sans-serif, system-ui"
            >
              {n.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
