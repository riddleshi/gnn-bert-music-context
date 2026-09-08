import { SlideDeck } from "@/components/SlideDeck";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Slides — Cadence Lab",
  description: "Course talk: GNN-based BERT for musical context, 8–10 minutes.",
};

export default function SlidesPage() {
  return <SlideDeck />;
}
