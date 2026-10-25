import type { AsrSettings } from "@/lib/settings";

/** The slice of OpenRouter's model object the UI reads. */
export type AsrModel = {
  id: string;
  name: string;
  description: string;
  hugging_face_id: string | null;
  languages: string[];
  loaded: boolean;
  /** OpenRouter's field: the request parameters, including tuning options, this model takes. */
  supported_parameters: string[];
  /** Whisper: the longest audio per pass (30 s). The server always chunks to it. */
  max_chunk_seconds: number | null;
  /** False when the server cannot fetch the checkpoint (e.g. no HF_TOKEN for a private repo). */
  available: boolean;
  unavailable_reason: string | null;
};

export type Segment = { id: number; start: number; end: number; text: string };
export type Word = { word: string; start: number; end: number };

/** OpenRouter's STT response. `segments`/`words` come with verbose_json. */
export type TranscriptionResponse = {
  text: string;
  usage: { seconds: number; cost: number };
  language?: string;
  duration?: number;
  segments?: Segment[];
  words?: Word[];
};

export type Health = {
  status: "ok";
  default_model: string;
  loaded_models: string[];
  busy: boolean;
  queue_length: number;
};

export type TranscriptionResult = TranscriptionResponse & {
  id: string;
  model: string;
  source: string;
  /** Timings from the X-*-Ms headers; null when the backend sent none. */
  timings: { totalMs: number; queueMs: number; loadMs: number; inferenceMs: number } | null;
  /** How many chunks the audio was transcribed in (1 = one pass). */
  chunks: number;
  /** The settings this run used, and what the model accepted of them. */
  settings: AsrSettings;
  supported: string[];
  maxChunkSeconds: number | null;
  createdAt: number;
};
