import type { NextRequest } from "next/server";

const API = process.env.CADENCE_API ?? "http://127.0.0.1:43181";

export async function POST(req: NextRequest) {
  const body = await req.text();
  try {
    const res = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body,
    });
    const text = await res.text();
    return new Response(text, {
      status: res.status,
      headers: { "content-type": res.headers.get("content-type") ?? "application/json" },
    });
  } catch {
    return Response.json(
      { error: "Python inference server is not running on port 43181." },
      { status: 503 },
    );
  }
}
