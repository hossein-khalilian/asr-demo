import type { AsrModel, Health, TranscriptionResponse, TranscriptionResult } from "@/lib/types";

/** An error body in OpenRouter's envelope: `{ error: { code, message } }`. */
export class ApiError extends Error {
  constructor(
    public readonly code: number,
    message: string,
  ) {
    super(message);
  }
}

async function parse<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const message = body?.error?.message ?? `Request failed with status ${res.status}`;
    throw new ApiError(body?.error?.code ?? res.status, message);
  }
  return body as T;
}

export async function fetchModels(): Promise<AsrModel[]> {
  const res = await fetch("/api/v1/models", { cache: "no-store" });
  return (await parse<{ data: AsrModel[] }>(res)).data;
}

export async function fetchHealth(): Promise<Health> {
  return parse<Health>(await fetch("/api/health", { cache: "no-store" }));
}

export type TranscribeOptions = {
  model: string;
  timestamps: boolean;
  signal?: AbortSignal;
};

export async function transcribe(
  audio: Blob,
  filename: string,
  { model, timestamps, signal }: TranscribeOptions,
): Promise<TranscriptionResult> {
  const form = new FormData();
  form.append("file", audio, filename);
  form.append("model", model);
  if (timestamps) {
    form.append("response_format", "verbose_json");
    form.append("timestamp_granularities[]", "segment");
    form.append("timestamp_granularities[]", "word");
  }
  const res = await fetch("/api/v1/audio/transcriptions", { method: "POST", body: form, signal });
  const body = await parse<TranscriptionResponse>(res);
  const processing = Number(res.headers.get("x-processing-ms"));
  return {
    ...body,
    id: res.headers.get("x-generation-id") ?? crypto.randomUUID(),
    model: res.headers.get("x-model") ?? model,
    source: filename,
    processingMs: Number.isFinite(processing) && processing > 0 ? processing : null,
    createdAt: Date.now(),
  };
}
