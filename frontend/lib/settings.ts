/**
 * Inference settings the UI can tune. They travel to the backend as
 * OpenRouter's `provider.options["asr-demo"]` (plus OpenRouter's own
 * `temperature`), and only the ones the chosen model lists in
 * `supported_parameters` are sent.
 */

import { useSyncExternalStore } from "react";

export type Decoder = "default" | "rnnt" | "ctc";

export type AsrSettings = {
  decoder: Decoder;
  beamSize: number;
  temperature: number;
  chunking: boolean;
  chunkSeconds: number;
  overlapSeconds: number;
  normalize: boolean;
  denoise: boolean;
  highpass: boolean;
};

/** The backend's defaults: a request with these is an untuned request. */
export const DEFAULT_SETTINGS: AsrSettings = {
  decoder: "default",
  beamSize: 1,
  temperature: 0,
  chunking: false,
  chunkSeconds: 30,
  overlapSeconds: 2,
  normalize: false,
  denoise: false,
  highpass: false,
};

export const PROVIDER_SLUG = "asr-demo";

/** Overlap must stay under half a chunk (the backend rejects anything else). */
export function maxOverlap(chunkSeconds: number): number {
  return Math.min(10, Math.floor((chunkSeconds / 2 - 0.5) * 2) / 2);
}

/**
 * `provider.options["asr-demo"]`: the non-default settings this model takes,
 * with the chunk length kept within the model's window (Whisper: 30 s).
 */
export function providerOptions(
  s: AsrSettings,
  supported: string[],
  maxChunkSeconds: number | null = null,
): Record<string, unknown> {
  const has = (name: string) => supported.includes(name);
  const options: Record<string, unknown> = {};
  if (has("decoder") && s.decoder !== "default") options.decoder = s.decoder;
  if (has("beam_size") && s.beamSize > 1) options.beam_size = s.beamSize;
  if (has("chunk_seconds") && s.chunking) {
    const chunk = maxChunkSeconds ? Math.min(s.chunkSeconds, maxChunkSeconds) : s.chunkSeconds;
    options.chunk_seconds = chunk;
    options.overlap_seconds = Math.min(s.overlapSeconds, maxOverlap(chunk));
  }
  for (const name of ["normalize", "denoise", "highpass"] as const) {
    if (has(name) && s[name]) options[name] = true;
  }
  return options;
}

/** Short labels for what a run changed, e.g. ["CTC", "beam 4", "chunks 30s/2s"]. */
export function describeSettings(
  s: AsrSettings,
  supported: string[],
  maxChunkSeconds: number | null = null,
): string[] {
  const options = providerOptions(s, supported, maxChunkSeconds);
  const labels: string[] = [];
  if (options.decoder) labels.push(String(options.decoder).toUpperCase());
  if (options.beam_size) labels.push(`beam ${options.beam_size}`);
  if (supported.includes("temperature") && s.temperature > 0) {
    labels.push(`temp ${s.temperature}`);
  }
  if (options.chunk_seconds) {
    labels.push(`chunks ${options.chunk_seconds}s/${options.overlap_seconds}s`);
  }
  if (options.normalize) labels.push("normalize");
  if (options.denoise) labels.push("denoise");
  if (options.highpass) labels.push("high-pass");
  return labels;
}

const STORAGE_KEY = "asr-demo:settings";

/** Settings are a per-browser convenience; storage may be unavailable. */
export function loadSettings(): AsrSettings {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
    return saved ? { ...DEFAULT_SETTINGS, ...saved } : DEFAULT_SETTINGS;
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(s: AsrSettings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
  } catch {
    // Private mode or blocked storage: settings just will not persist.
  }
}

// A tiny store over localStorage. useSyncExternalStore renders the defaults on
// the server and during hydration, then the saved settings, so the HTML the
// server sent never mismatches.
const listeners = new Set<() => void>();
let current: AsrSettings | null = null;

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function snapshot(): AsrSettings {
  current ??= loadSettings();
  return current;
}

export function setSettings(next: AsrSettings): void {
  current = next;
  saveSettings(next);
  listeners.forEach((listener) => listener());
}

/**
 * Merge ``patch`` into the latest settings. Controls send patches rather than
 * whole objects so two quick changes cannot overwrite each other from a stale
 * render.
 */
export function updateSettings(patch: Partial<AsrSettings>): void {
  setSettings({ ...snapshot(), ...patch });
}

export function useSettings(): AsrSettings {
  return useSyncExternalStore(subscribe, snapshot, () => DEFAULT_SETTINGS);
}
