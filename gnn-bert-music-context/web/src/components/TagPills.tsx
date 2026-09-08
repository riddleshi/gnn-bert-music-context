"use client";

import { Badge } from "@/components/ui/badge";
import type { TagScore } from "@/lib/types";

export function TagPills({
  tags,
  scores,
  truth,
}: {
  tags: string[];
  scores?: TagScore[];
  truth?: string[];
}) {
  const truthSet = new Set(truth ?? []);
  const scoreMap = new Map((scores ?? []).map((s) => [s.tag, s.score]));
  const ordered = scores?.length ? scores.map((s) => s.tag) : tags;
  if (!ordered.length) {
    return <p className="text-xs text-mute">No tags above threshold.</p>;
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {ordered.map((tag) => {
        const hit = truthSet.size ? truthSet.has(tag) : tags.includes(tag);
        const sc = scoreMap.get(tag);
        return (
          <Badge key={tag} tone={hit ? "ok" : "default"}>
            {tag.replaceAll("_", " ")}
            {sc != null ? ` ${sc.toFixed(2)}` : ""}
          </Badge>
        );
      })}
    </div>
  );
}
