import "server-only";

/** Where the backend lives, as seen from the Next.js server (not the browser). */
export const API_URL = (process.env.API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

export function backendError(code: number, message: string): Response {
  return Response.json({ error: { code, message } }, { status: code });
}
