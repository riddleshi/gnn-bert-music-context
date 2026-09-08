const PAGE_COUNT = 6;
const PAPER_REV = "6p-584390";

const PAGES = Array.from(
  { length: PAGE_COUNT },
  (_, i) => `/paper/page-${String(i + 1).padStart(2, "0")}.png?v=${PAPER_REV}`,
);

export function PaperView() {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="font-serif text-3xl">Course paper</h2>
          <p className="mt-2 max-w-2xl text-mute">
            6-page IEEE conference paper (IEEEtran, two-column). Source is{" "}
            <code className="text-cream">report/final_report.tex</code>; download the PDF if you want the
            typeset file.
          </p>
        </div>
        <a
          href={`/paper/final_report.pdf?v=${PAPER_REV}`}
          className="rounded-full bg-copper px-4 py-2 text-sm font-medium text-ink hover:bg-copper/90"
        >
          Download PDF
        </a>
      </div>
      <div className="space-y-4">
        {PAGES.map((src, i) => (
          <figure key={src} className="overflow-hidden rounded-2xl border border-line bg-white">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={src} alt={`Report page ${i + 1}`} className="w-full" />
            <figcaption className="bg-panel px-4 py-2 text-center text-xs text-mute">
              Page {i + 1} / {PAGES.length}
            </figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
