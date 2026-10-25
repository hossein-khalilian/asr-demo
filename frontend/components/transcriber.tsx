"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, Mic, Upload } from "lucide-react";
import { RecordPanel } from "@/components/record-panel";
import { ResultCard, type ResultWithAudio } from "@/components/result-card";
import { UploadPanel } from "@/components/upload-panel";
import { fetchHealth, fetchModels, transcribe } from "@/lib/api";
import type { AsrModel } from "@/lib/types";
import { cn } from "@/lib/utils";

type Tab = "record" | "upload";

export function Transcriber() {
  const [models, setModels] = useState<AsrModel[]>([]);
  const [model, setModel] = useState("");
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("record");
  const [timestamps, setTimestamps] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ResultWithAudio[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const resultsRef = useRef(results);

  useEffect(() => {
    resultsRef.current = results;
  }, [results]);

  useEffect(() => {
    Promise.all([fetchModels(), fetchHealth().catch(() => null)])
      .then(([list, health]) => {
        setModels(list);
        // Start on the server's default model (ASR_DEFAULT_MODEL), not just the first listed.
        const usable = list.filter((m) => m.available);
        const preferred = usable.find((m) => m.id === health?.default_model) ?? usable[0];
        setModel((current) => current || preferred?.id || "");
      })
      .catch((err: Error) => setModelsError(err.message));
  }, []);

  // Release the audio blobs the results keep for playback.
  useEffect(() => () => resultsRef.current.forEach((r) => URL.revokeObjectURL(r.audioUrl)), []);

  const submit = async (audio: Blob, filename: string) => {
    if (!model) return;
    setError(null);
    setPending(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const result = await transcribe(audio, filename, {
        model,
        timestamps,
        signal: controller.signal,
      });
      setResults((prev) => [{ ...result, audioUrl: URL.createObjectURL(audio) }, ...prev]);
    } catch (err) {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : String(err));
    } finally {
      abortRef.current = null;
      setPending(false);
    }
  };

  const remove = (id: string) =>
    setResults((prev) => {
      const gone = prev.find((r) => r.id === id);
      if (gone) URL.revokeObjectURL(gone.audioUrl);
      return prev.filter((r) => r.id !== id);
    });

  const selected = models.find((m) => m.id === model);

  return (
    <div className="flex flex-col gap-6">
      <section className="rounded-2xl border border-border bg-card p-5 shadow-sm sm:p-6">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
          <label className="flex min-w-0 flex-col gap-1.5">
            <span className="text-sm font-medium">Model</span>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={pending || models.length === 0}
              className="w-full min-w-0 rounded-xl border border-border bg-background px-3 py-2.5 text-sm outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
            >
              {models.length === 0 && <option>{modelsError ? "Unavailable" : "Loading…"}</option>}
              {models.map((m) => (
                <option key={m.id} value={m.id} disabled={!m.available}>
                  {m.name}
                  {m.loaded ? "  ●" : ""}
                  {m.available ? "" : " (unavailable)"}
                </option>
              ))}
            </select>
          </label>
          <label className="flex cursor-pointer items-center gap-2 pb-2.5 text-sm">
            <input
              type="checkbox"
              checked={timestamps}
              onChange={(e) => setTimestamps(e.target.checked)}
              className="size-4 accent-[oklch(var(--primary))]"
            />
            Timestamps
          </label>
        </div>
        {selected && (
          <p className="mt-2 text-xs text-muted-foreground">
            {selected.description}{" "}
            {selected.hugging_face_id && (
              <a
                href={`https://huggingface.co/${selected.hugging_face_id}`}
                target="_blank"
                rel="noreferrer"
                className="font-mono underline-offset-2 hover:underline"
              >
                {selected.hugging_face_id}
              </a>
            )}
            {selected.available &&
              !selected.loaded &&
              " · loads on first use, which can take a minute"}
          </p>
        )}
        {selected && !selected.available && (
          <p className="mt-2 rounded-lg bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-400">
            Unavailable on this server: {selected.unavailable_reason}.
          </p>
        )}
        {modelsError && (
          <p className="mt-2 text-sm text-destructive">Could not load models: {modelsError}</p>
        )}

        <div
          role="tablist"
          className="mt-6 grid grid-cols-2 gap-1 rounded-xl bg-muted p-1 text-sm font-medium"
        >
          {(
            [
              ["record", "Record", Mic],
              ["upload", "Upload", Upload],
            ] as const
          ).map(([id, label, Icon]) => (
            <button
              key={id}
              role="tab"
              type="button"
              aria-selected={tab === id}
              onClick={() => setTab(id)}
              className={cn(
                "flex items-center justify-center gap-2 rounded-lg py-2 transition",
                tab === id ? "bg-card shadow-sm" : "text-muted-foreground hover:text-foreground",
              )}
            >
              <Icon className="size-4" />
              {label}
            </button>
          ))}
        </div>

        {tab === "record" ? (
          <RecordPanel disabled={pending || !model} onRecorded={submit} />
        ) : (
          <UploadPanel disabled={pending || !model} onSubmit={submit} />
        )}

        {pending && (
          <div className="flex items-center justify-center gap-3 rounded-xl bg-muted/60 py-3 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Transcribing…
            <button
              type="button"
              onClick={() => abortRef.current?.abort()}
              className="underline-offset-2 hover:text-foreground hover:underline"
            >
              Cancel
            </button>
          </div>
        )}
        {error && (
          <p className="rounded-xl bg-destructive/10 px-4 py-3 text-sm text-destructive">{error}</p>
        )}
      </section>

      {results.map((result) => (
        <ResultCard key={result.id} result={result} onRemove={() => remove(result.id)} />
      ))}
    </div>
  );
}
