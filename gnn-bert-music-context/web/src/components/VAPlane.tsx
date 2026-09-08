"use client";

import { clamp01 } from "@/lib/utils";

export function VAPlane({
  valence,
  arousal,
  predValence,
  predArousal,
}: {
  valence: number;
  arousal: number;
  predValence?: number;
  predArousal?: number;
}) {
  const x = clamp01(valence);
  const y = 1 - clamp01(arousal);
  const px = predValence != null ? clamp01(predValence) : null;
  const py = predArousal != null ? 1 - clamp01(predArousal) : null;
  return (
    <div className="relative aspect-square w-full overflow-hidden rounded-xl border border-line bg-ink">
      <div className="absolute left-1/2 top-0 h-full w-px bg-white/10" />
      <div className="absolute left-0 top-1/2 h-px w-full bg-white/10" />
      <span className="absolute left-2 top-2 text-[10px] uppercase tracking-widest text-mute">High arousal</span>
      <span className="absolute bottom-2 left-2 text-[10px] uppercase tracking-widest text-mute">Low valence</span>
      <span className="absolute bottom-2 right-2 text-[10px] uppercase tracking-widest text-mute">High valence</span>
      <span
        className="absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-copper"
        style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
        title="true"
      />
      {px != null && py != null ? (
        <span
          className="absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-teal"
          style={{ left: `${px * 100}%`, top: `${py * 100}%` }}
          title="predicted"
        />
      ) : null}
    </div>
  );
}
