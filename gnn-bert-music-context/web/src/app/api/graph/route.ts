import type { NextRequest } from "next/server";

const API = process.env.CADENCE_API ?? "http://127.0.0.1:43181";

export async function GET(req: NextRequest) {
  const file = req.nextUrl.searchParams.get("file") ?? "";
  try {
    const res = await fetch(`${API}/graph/${encodeURIComponent(file)}`);
    const text = await res.text();
    return new Response(text, {
      status: res.status,
      headers: { "content-type": res.headers.get("content-type") ?? "application/json" },
    });
  } catch {
    return Response.json(
      { error: "Python API is not running. Start: python -m uvicorn api.server:app --port 43181" },
      { status: 503 },
    );
  }
}
