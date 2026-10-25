import type { NextRequest } from "next/server";
import { fetch, type BodyInit } from "undici";
import { API_URL, backendAgent, proxyFailure } from "@/lib/server";

// Same-origin proxy to the backend's OpenRouter-style API. The browser never
// talks to the backend directly, so there is no CORS to configure and the
// backend's API key stays on the server.

export const dynamic = "force-dynamic";

const REQUEST_HEADERS = ["accept", "content-type", "content-length"];
const RESPONSE_HEADERS = [
  "content-type",
  "retry-after",
  "x-generation-id",
  "x-model",
  "x-processing-ms",
  "x-queue-ms",
  "x-load-ms",
  "x-inference-ms",
  "x-chunks",
];

type Context = { params: Promise<{ path: string[] }> };

async function proxy(request: NextRequest, { params }: Context): Promise<Response> {
  const { path } = await params;
  const target = `${API_URL}/api/v1/${path.map(encodeURIComponent).join("/")}${request.nextUrl.search}`;

  const headers = new Headers();
  for (const name of REQUEST_HEADERS) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  // A caller with its own key (an API client using this host as its base URL)
  // keeps it; the browser UI sends none and gets the server's.
  const callerAuth = request.headers.get("authorization");
  const apiKey = process.env.ASR_API_KEY?.trim();
  if (callerAuth) headers.set("authorization", callerAuth);
  else if (apiKey) headers.set("authorization", `Bearer ${apiKey}`);

  let upstream: Awaited<ReturnType<typeof fetch>>;
  try {
    upstream = await fetch(target, {
      method: request.method,
      headers,
      // Stream the upload through rather than buffering it in memory.
      // The DOM's and undici's ReadableStream types differ; the object is the same.
      body: request.method === "GET" ? undefined : (request.body as unknown as BodyInit),
      duplex: "half",
      dispatcher: backendAgent,
    });
  } catch (error) {
    return proxyFailure(error);
  }

  const responseHeaders = new Headers();
  for (const name of RESPONSE_HEADERS) {
    const value = upstream.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }
  return new Response(upstream.body as ReadableStream | null, {
    status: upstream.status,
    headers: responseHeaders,
  });
}

export { proxy as GET, proxy as POST };
