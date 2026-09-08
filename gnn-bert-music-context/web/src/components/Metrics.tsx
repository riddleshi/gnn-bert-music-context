"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmt } from "@/lib/utils";
import type { DemoPayload } from "@/lib/types";

const MODEL_ROWS = [
  ["random", "Random tags"],
  ["majority", "Majority"],
  ["cnn_mel", "CNN mel-spec"],
  ["bert_only", "Task 1 · BERT"],
  ["gnn_only", "Task 2 · GNN"],
  ["fusion_concat", "Task 3 · concat"],
  ["fusion_cross_attention", "Task 3 · cross-attn"],
] as const;

export function MetricsTable({ demo }: { demo: DemoPayload }) {
  const models = demo.metrics.models;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-[11px] uppercase tracking-widest text-mute">
          <tr>
            <th className="pb-3 font-medium">Model</th>
            <th className="pb-3 font-medium">Macro-F1</th>
            <th className="pb-3 font-medium">Micro-F1</th>
            <th className="pb-3 font-medium">AUC-PR</th>
            <th className="pb-3 font-medium">MAE emo</th>
            <th className="pb-3 font-medium">Genre acc</th>
          </tr>
        </thead>
        <tbody>
          {MODEL_ROWS.map(([key, label]) => {
            const m = models[key] ?? {};
            return (
              <tr key={key} className="border-t border-line/70">
                <td className="py-2.5 text-cream">{label}</td>
                <td className="py-2.5 font-mono text-copper">{fmt(m.macro_f1 ?? NaN)}</td>
                <td className="py-2.5 font-mono">{fmt(m.micro_f1 ?? NaN)}</td>
                <td className="py-2.5 font-mono">{fmt(m.auc_pr ?? NaN)}</td>
                <td className="py-2.5 font-mono">{fmt(m.mae_emotion ?? NaN)}</td>
                <td className="py-2.5 font-mono">{fmt(m.genre_acc ?? NaN)}</td>
              </tr>
            );
          })}
          <tr className="border-t border-line/70">
            <td className="py-2.5 text-cream">Task 4 · contrastive R@5</td>
            <td className="py-2.5 font-mono text-teal" colSpan={5}>
              caption→audio {fmt(models.contrastive?.["caption_to_audio_R@5"] ?? NaN)} · audio→caption{" "}
              {fmt(models.contrastive?.["audio_to_caption_R@5"] ?? NaN)}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export function AblationChart({ demo }: { demo: DemoPayload }) {
  const data = MODEL_ROWS.map(([key, label]) => ({
    name: label.replace("Task 3 · ", "").replace("Task 2 · ", "").replace("Task 1 · ", ""),
    f1: Number(demo.metrics.models[key]?.macro_f1 ?? 0),
    auc: Number(demo.metrics.models[key]?.auc_pr ?? 0),
  }));
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 24 }}>
          <CartesianGrid stroke="#2a241c" vertical={false} />
          <XAxis dataKey="name" tick={{ fill: "#b7aa98", fontSize: 11 }} interval={0} angle={-18} textAnchor="end" height={50} />
          <YAxis tick={{ fill: "#b7aa98", fontSize: 11 }} domain={[0, 1]} />
          <Tooltip contentStyle={{ background: "#16130f", border: "1px solid #3a3228", color: "#efe6d6" }} />
          <Legend />
          <Bar dataKey="f1" name="Macro-F1" fill="#c45c26" radius={[6, 6, 0, 0]} />
          <Bar dataKey="auc" name="AUC-PR" fill="#1f8a7a" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function F1Curves({ demo }: { demo: DemoPayload }) {
  const keys = [
    ["task1", "BERT"],
    ["task2_tags", "GNN"],
    ["cnn_tags", "CNN"],
    ["task3_concat", "Concat"],
    ["task3_cross_attention", "Cross-attn"],
  ] as const;
  const maxLen = Math.max(0, ...keys.map(([k]) => demo.history[k]?.length ?? 0));
  const data = Array.from({ length: maxLen }, (_, i) => {
    const row: Record<string, number> = { epoch: i + 1 };
    for (const [k, label] of keys) {
      const v = demo.history[k]?.[i]?.val_macro_f1;
      if (typeof v === "number") row[label] = v;
    }
    return row;
  });
  const colors: Record<string, string> = {
    BERT: "#c9a227",
    GNN: "#5ee0c0",
    CNN: "#8a7a68",
    Concat: "#6b8cae",
    "Cross-attn": "#c45c26",
  };
  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#2a241c" vertical={false} />
          <XAxis dataKey="epoch" tick={{ fill: "#b7aa98", fontSize: 11 }} />
          <YAxis tick={{ fill: "#b7aa98", fontSize: 11 }} domain={[0, 1]} />
          <Tooltip contentStyle={{ background: "#16130f", border: "1px solid #3a3228", color: "#efe6d6" }} />
          <Legend />
          {keys.map(([, label]) => (
            <Line key={label} type="monotone" dataKey={label} stroke={colors[label]} strokeWidth={2} dot={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
