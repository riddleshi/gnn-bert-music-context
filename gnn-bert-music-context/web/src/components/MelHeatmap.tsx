"use client";

export function MelHeatmap({
  data,
  className,
}: {
  data: number[][];
  className?: string;
}) {
  const rows = data.length;
  const cols = data[0]?.length ?? 0;
  if (!rows || !cols) return null;
  let min = Infinity;
  let max = -Infinity;
  for (const row of data) {
    for (const v of row) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const cells: string[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const t = (data[r][c] - min) / (max - min + 1e-6);
      const hue = 28 + t * 18;
      const light = 12 + t * 42;
      cells.push(`<rect x="${c}" y="${r}" width="1" height="1" fill="hsl(${hue} 70% ${light}%)" />`);
    }
  }
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${cols} ${rows}" preserveAspectRatio="none">${cells.join("")}</svg>`;
  const uri = `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
  return (
    <img
      src={uri}
      alt="Log-mel spectrogram"
      className={className ?? "h-40 w-full rounded-xl object-cover"}
    />
  );
}
