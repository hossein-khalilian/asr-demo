import "server-only";
import { Agent } from "undici";

/** Where the backend lives, as seen from the Next.js server (not the browser). */
export const API_URL = (process.env.API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

/**
 * How long the proxy waits for the backend. Node's fetch gives up on response
 * headers after 5 minutes, but transcription on CPU, with beam search or a
 * queue in front, can take longer than that.
 */
export const PROXY_TIMEOUT_SECONDS = Number(process.env.ASR_PROXY_TIMEOUT_SECONDS) || 30 * 60;

export const backendAgent = new Agent({
  headersTimeout: PROXY_TIMEOUT_SECONDS * 1000,
  bodyTimeout: PROXY_TIMEOUT_SECONDS * 1000,
});

export function backendError(code: number, message: string): Response {
  return Response.json({ error: { code, message } }, { status: code });
}

/** A 504 when the backend took too long, a 502 when it could not be reached. */
export function proxyFailure(error: unknown): Response {
  const code = (error as { cause?: { code?: string } })?.cause?.code ?? "";
  if (code === "UND_ERR_HEADERS_TIMEOUT" || code === "UND_ERR_BODY_TIMEOUT") {
    return backendError(
      504,
      `The ASR backend did not answer within ${Math.round(PROXY_TIMEOUT_SECONDS / 60)} minutes.`,
    );
  }
  return backendError(502, "The ASR backend is unreachable.");
}
