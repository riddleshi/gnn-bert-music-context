import { PrintButton } from "@/components/PrintButton";
import { SLIDES, TALK_COURSE, TALK_TARGET, TALK_TITLE, TOTAL_SEC, scriptWordCount } from "@/lib/presentation";
import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Speaker script — Cadence Lab",
  description: "Timed 8–10 minute script for the course slide deck.",
};

function fmtTime(sec: number) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export default function ScriptPage() {
  const words = scriptWordCount();
  let t = 0;
  return (
    <div className="min-h-screen bg-ink text-cream">
      <header className="border-b border-line px-4 py-3 text-xs text-mute print:hidden">
        <div className="mx-auto flex max-w-3xl items-center justify-between">
          <Link href="/slides" className="hover:text-cream">
            ← Slides
          </Link>
          <div className="flex items-center gap-3">
            <a href="/cadence-lab-slides.pptx" download className="rounded-full border border-line px-3 py-1 hover:text-cream">
              PowerPoint
            </a>
            <PrintButton />
          </div>
        </div>
      </header>
      <article className="mx-auto max-w-3xl px-4 py-12 print:max-w-none print:px-0 print:py-0">
        <p className="text-[11px] uppercase tracking-[0.25em] text-copper">{TALK_COURSE}</p>
        <h1 className="mt-3 font-serif text-4xl leading-tight">{TALK_TITLE}</h1>
        <p className="mt-4 text-mute">
          Speaker script · {TALK_TARGET} · {SLIDES.length} slides · {words} words
          (~{Math.round((words / 140) * 10) / 10}–{Math.round((words / 130) * 10) / 10} min at 130–140 wpm).
          Slide clock {fmtTime(TOTAL_SEC)} includes pauses. Open notes with <kbd className="text-cream">N</kbd>.
        </p>
        <p className="mt-2 text-sm text-mute">
          Pace: start each slide at the time on the left. If you run long on Data or Reading the
          table, cut the case-study names and jump to takeaways.
        </p>
        <ol className="mt-10 space-y-10">
          {SLIDES.map((slide, idx) => {
            const start = t;
            t += slide.durationSec;
            return (
              <li key={slide.id} className="break-inside-avoid">
                <p className="text-[11px] uppercase tracking-widest text-copper">
                  {fmtTime(start)}–{fmtTime(t)} · slide {idx + 1}/{SLIDES.length} · {slide.kicker}
                </p>
                <h2 className="mt-1 font-serif text-2xl">
                  <Link href={`/slides#${idx + 1}`} className="hover:text-copper">
                    {slide.title}
                  </Link>
                </h2>
                <p className="mt-3 text-[17px] leading-relaxed text-cream/90">{slide.script}</p>
              </li>
            );
          })}
        </ol>
        <p className="mt-12 text-sm text-mute">
          End by {fmtTime(TOTAL_SEC)}. Hold remaining time for questions. Numbers match{" "}
          <code className="text-cream">results/metrics.json</code> and the IEEE paper.
        </p>
      </article>
    </div>
  );
}
