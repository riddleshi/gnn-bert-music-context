"use client";

export function PrintButton() {
  return (
    <button
      type="button"
      className="rounded-full border border-line px-3 py-1 hover:text-cream"
      onClick={() => window.print()}
    >
      Print
    </button>
  );
}
