import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Badge({
  className,
  tone = "default",
  children,
}: {
  className?: string;
  tone?: "default" | "copper" | "teal" | "ok" | "warn";
  children: ReactNode;
}) {
  const tones = {
    default: "border-line text-cream/80",
    copper: "border-copper/40 text-copper bg-copper/10",
    teal: "border-teal/40 text-teal bg-teal/10",
    ok: "border-emerald-400/40 text-emerald-300 bg-emerald-400/10",
    warn: "border-amber-400/40 text-amber-200 bg-amber-400/10",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] tracking-wide uppercase",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
