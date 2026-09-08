"use client";

import { SLIDES, TOTAL_SEC, scriptWordCount } from "@/lib/presentation";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

function fmtTime(sec: number) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function SlideDeck() {
  const [i, setI] = useState(0);
  const [notes, setNotes] = useState(false);
  const [help, setHelp] = useState(false);
  const slide = SLIDES[i];
  const words = useMemo(() => scriptWordCount(), []);
  const elapsed = SLIDES.slice(0, i).reduce((s, d) => s + d.durationSec, 0);

  const go = useCallback((n: number) => {
    setI(Math.max(0, Math.min(SLIDES.length - 1, n)));
  }, []);

  useEffect(() => {
    const fromHash = Number.parseInt(window.location.hash.replace("#", ""), 10);
    if (Number.isFinite(fromHash) && fromHash >= 1 && fromHash <= SLIDES.length) {
      setI(fromHash - 1);
    }
  }, []);

  useEffect(() => {
    window.history.replaceState(null, "", `#${i + 1}`);
  }, [i]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      if (e.key === "ArrowRight" || e.key === " " || e.key === "PageDown") {
        e.preventDefault();
        go(i + 1);
      } else if (e.key === "ArrowLeft" || e.key === "PageUp") {
        e.preventDefault();
        go(i - 1);
      } else if (e.key === "Home") go(0);
      else if (e.key === "End") go(SLIDES.length - 1);
      else if (e.key === "n" || e.key === "N") setNotes((v) => !v);
      else if (e.key === "?" || e.key === "h" || e.key === "H") setHelp((v) => !v);
      else if (e.key === "Escape") {
        setHelp(false);
        setNotes(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [go, i]);

  return (
    <div className="flex min-h-screen flex-col bg-ink text-cream">
      <header className="flex items-center justify-between gap-3 border-b border-line/80 px-4 py-2 text-xs text-mute">
        <div className="flex items-center gap-3">
          <Link href="/" className="hover:text-cream">
            Cadence Lab
          </Link>
          <span className="text-line">/</span>
          <span>Slides</span>
        </div>
        <div className="flex items-center gap-2">
          <Link href="/slides/script" className="rounded-full border border-line px-3 py-1 hover:text-cream">
            Speaker script
          </Link>
          <a
            href="/cadence-lab-slides.pptx"
            download
            className="rounded-full bg-copper px-3 py-1 font-medium text-ink hover:bg-copper/90"
          >
            Download PowerPoint
          </a>
          <button
            type="button"
            onClick={() => setNotes((v) => !v)}
            className={cn(
              "rounded-full border px-3 py-1",
              notes ? "border-copper text-copper" : "border-line hover:text-cream",
            )}
          >
            Notes (N)
          </button>
        </div>
      </header>

      <div className="h-0.5 bg-line">
        <div
          className="h-full bg-copper transition-[width] duration-300"
          style={{ width: `${((i + 1) / SLIDES.length) * 100}%` }}
        />
      </div>

      <main className="relative mx-auto flex w-full max-w-6xl flex-1 flex-col px-6 py-8 md:px-10 md:py-12">
        <p className="text-[11px] uppercase tracking-[0.28em] text-copper">{slide.kicker}</p>
        <h1
          className={cn(
            "mt-3 font-serif leading-tight text-cream",
            slide.layout === "title" ? "text-4xl md:text-6xl" : "text-3xl md:text-5xl",
          )}
        >
          {slide.title}
        </h1>

        <div className="mt-8 flex-1">
          {slide.layout === "title" ? (
            <ul className="mt-6 space-y-3 text-lg text-mute md:text-2xl">
              {slide.bullets?.map((b) => (
                <li key={b} className="border-l-2 border-copper/70 pl-4">
                  {b}
                </li>
              ))}
            </ul>
          ) : null}

          {slide.layout === "bullets" ? (
            <ul className="space-y-4 text-lg text-cream/90 md:text-2xl">
              {slide.bullets?.map((b) => (
                <li key={b} className="flex gap-3">
                  <span className="mt-2 h-2 w-2 shrink-0 rounded-full bg-copper" />
                  <span>{b}</span>
                </li>
              ))}
            </ul>
          ) : null}

          {slide.layout === "metrics" ? (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {slide.metrics?.map((m) => (
                <div key={m.label} className="rounded-2xl border border-line bg-panel p-5">
                  <p className="text-[11px] uppercase tracking-widest text-mute">{m.label}</p>
                  <p className="mt-2 font-serif text-4xl text-copper md:text-5xl">{m.value}</p>
                  {m.hint ? <p className="mt-2 text-sm text-mute">{m.hint}</p> : null}
                </div>
              ))}
            </div>
          ) : null}

          {slide.layout === "split" ? (
            <div
              className={cn(
                "grid gap-4",
                (slide.split?.length ?? 0) === 2
                  ? "md:grid-cols-2"
                  : (slide.split?.length ?? 0) >= 4
                    ? "sm:grid-cols-2 lg:grid-cols-4"
                    : "md:grid-cols-3",
              )}
            >
              {slide.split?.map((col) => (
                <div key={col.heading} className="rounded-2xl border border-line bg-panel p-5">
                  <h2 className="font-serif text-xl text-copper">{col.heading}</h2>
                  <ul className="mt-4 space-y-2 text-sm text-cream/90 md:text-base">
                    {col.items.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          ) : null}

          {slide.layout === "close" ? (
            <ul className="space-y-4 text-lg text-cream/90 md:text-2xl">
              {slide.bullets?.map((b) => (
                <li key={b} className="border-l-2 border-teal/70 pl-4">
                  {b}
                </li>
              ))}
            </ul>
          ) : null}

          {slide.footnote ? <p className="mt-8 text-sm text-mute">{slide.footnote}</p> : null}
        </div>
      </main>

      {notes ? (
        <aside className="border-t border-line bg-panel px-6 py-4 md:px-10">
          <div className="mx-auto flex max-w-6xl flex-col gap-2">
            <p className="text-[11px] uppercase tracking-widest text-copper">
              Speak this slide · {fmtTime(slide.durationSec)} · cumulative {fmtTime(elapsed)}–{fmtTime(elapsed + slide.durationSec)}
            </p>
            <p className="max-w-4xl text-sm leading-relaxed text-cream/90 md:text-base">{slide.script}</p>
          </div>
        </aside>
      ) : null}

      <footer className="flex items-center justify-between gap-3 border-t border-line px-4 py-3 text-xs text-mute">
        <button type="button" className="rounded-full border border-line px-3 py-1 hover:text-cream" onClick={() => go(i - 1)}>
          ← Prev
        </button>
        <div className="flex items-center gap-3">
          <span>
            {i + 1} / {SLIDES.length}
          </span>
          <span className="hidden sm:inline">
            {fmtTime(TOTAL_SEC)} script · {words} words · ~{Math.round(words / 140)}–{Math.round(words / 125)} min
          </span>
          <button type="button" className="hover:text-cream" onClick={() => setHelp(true)}>
            Keys (?)
          </button>
        </div>
        <button type="button" className="rounded-full bg-copper px-3 py-1 text-ink hover:bg-copper/90" onClick={() => go(i + 1)}>
          Next →
        </button>
      </footer>

      {help ? (
        <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 p-4 md:items-center" onClick={() => setHelp(false)}>
          <div
            className="w-full max-w-md rounded-2xl border border-line bg-panel p-6 text-sm text-mute"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="font-serif text-xl text-cream">Keyboard</h2>
            <ul className="mt-4 space-y-2">
              <li>← → Space — previous / next</li>
              <li>Home / End — first / last</li>
              <li>N — speaker notes for this slide</li>
              <li>? — this help</li>
            </ul>
            <p className="mt-4">
              Full script: <Link href="/slides/script" className="text-copper hover:underline">/slides/script</Link>
            </p>
          </div>
        </div>
      ) : null}
    </div>
  );
}
