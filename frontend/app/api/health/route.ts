import { API_URL, backendError } from "@/lib/server";

export const dynamic = "force-dynamic";

export async function GET(): Promise<Response> {
  try {
    const upstream = await fetch(`${API_URL}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(5000),
    });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: { "content-type": "application/json" },
    });
  } catch {
    return backendError(502, "The ASR backend is unreachable.");
  }
}
