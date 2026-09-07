import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fmt(n: number, digits = 3) {
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

export function clamp01(n: number) {
  return Math.min(1, Math.max(0, n));
}
