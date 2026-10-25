import { PROVIDER_SLUG, providerOptions, type AsrSettings } from "@/lib/settings";
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
  settings: AsrSettings;
  /** The model's `supported_parameters`: only those settings are sent. */
  supported: string[];
  maxChunkSeconds: number | null;
  signal?: AbortSignal;
};

export async function transcribe(
  audio: Blob,
  filename: string,
  { model, timestamps, settings, supported, maxChunkSeconds, signal }: TranscribeOptions,
): Promise<TranscriptionResult> {
  const form = new FormData();
  form.append("file", audio, filename);
  form.append("model", model);
  if (timestamps) {
    form.append("response_format", "verbose_json");
    form.append("timestamp_granularities[]", "segment");
    form.append("timestamp_granularities[]", "word");
  }
  const options = providerOptions(settings, supported, maxChunkSeconds);
  if (Object.keys(options).length > 0) {
    // OpenRouter's provider preferences; object fields are JSON in multipart.
    form.append("provider", JSON.stringify({ options: { [PROVIDER_SLUG]: options } }));
  }
  if (supported.includes("temperature") && settings.temperature > 0) {
    form.append("temperature", String(settings.temperature));
  }
  const res = await fetch("/api/v1/audio/transcriptions", { method: "POST", body: form, signal });
  const body = await parse<TranscriptionResponse>(res);
  const ms = (name: string) => Number(res.headers.get(`x-${name}-ms`) ?? NaN);
  const timings = {
    totalMs: ms("processing"),
    queueMs: ms("queue"),
    loadMs: ms("load"),
    inferenceMs: ms("inference"),
  };
  return {
    ...body,
    id: res.headers.get("x-generation-id") ?? crypto.randomUUID(),
    model: res.headers.get("x-model") ?? model,
    source: filename,
    timings: Object.values(timings).every(Number.isFinite) ? timings : null,
    chunks: Number(res.headers.get("x-chunks")) || 1,
    settings,
    supported,
    maxChunkSeconds,
    createdAt: Date.now(),
  };
}
