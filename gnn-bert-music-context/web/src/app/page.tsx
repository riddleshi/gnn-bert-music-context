import { readFile } from "node:fs/promises";
import path from "node:path";
import { LabApp } from "./lab-app";
import type { DemoPayload } from "@/lib/types";

async function loadDemo(): Promise<DemoPayload | null> {
  try {
    const file = path.join(process.cwd(), "public", "data", "demo.json");
    return JSON.parse(await readFile(file, "utf8")) as DemoPayload;
  } catch {
    return null;
  }
}

export default async function Page() {
  const demo = await loadDemo();
  return <LabApp demo={demo} />;
}
