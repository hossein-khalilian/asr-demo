"use client";

import { useRef, useState } from "react";
import { Check, Copy, Trash2 } from "lucide-react";
import type { TranscriptionResult } from "@/lib/types";
import { formatClock, formatDuration } from "@/lib/utils";

export type ResultWithAudio = TranscriptionResult & { audioUrl: string };

export function ResultCard({
  result,
  onRemove,
}: {
  result: ResultWithAudio;
  onRemove: () => void;
}) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [copied, setCopied] = useState(false);
  const seconds = result.usage.seconds;
  const timings = result.timings;
  // Waiting for the model and loading it say nothing about its speed.
  const waitedMs = timings ? timings.queueMs + timings.loadMs : 0;

  const copy = async () => {
    await navigator.clipboard.writeText(result.text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const seek = (time: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = time;
    void audio.play();
  };

  return (
    <article className="rounded-2xl border border-border bg-card p-5 shadow-sm">
      <header className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
        <span className="font-mono">{result.model}</span>
        <span>·</span>
        <span>{result.source}</span>
        <span>·</span>
        <span>{formatDuration(seconds)} audio</span>
        {timings && (
          <>
            <span>·</span>
            <span
              title={[
                `Real-time factor ${(timings.inferenceMs / 1000 / seconds).toFixed(2)}`,
                `Queued ${formatDuration(timings.queueMs / 1000)}`,
                `Model load ${formatDuration(timings.loadMs / 1000)}`,
                `Total ${formatDuration(timings.totalMs / 1000)}`,
              ].join(" · ")}
            >
              {formatDuration(timings.inferenceMs / 1000)} to transcribe
              {waitedMs >= 1000 && ` (+${formatDuration(waitedMs / 1000)} waiting)`}
            </span>
          </>
        )}
        <span className="ms-auto flex gap-1">
          <button
            type="button"
            onClick={copy}
            className="rounded-md p-1.5 hover:bg-muted hover:text-foreground"
            aria-label="Copy transcript"
          >
            {copied ? <Check className="size-4 text-success" /> : <Copy className="size-4" />}
          </button>
          <button
            type="button"
            onClick={onRemove}
            className="rounded-md p-1.5 hover:bg-muted hover:text-foreground"
            aria-label="Remove result"
          >
            <Trash2 className="size-4" />
          </button>
        </span>
      </header>

      {result.text ? (
        <p dir="auto" className="whitespace-pre-wrap font-fa text-lg leading-loose">
          {result.text}
        </p>
      ) : (
        <p className="text-sm italic text-muted-foreground">No speech was recognised.</p>
      )}

      <audio ref={audioRef} src={result.audioUrl} controls className="mt-4 h-9 w-full" />

      {result.segments && result.segments.length > 1 && (
        <ol className="mt-4 divide-y divide-border rounded-xl border border-border">
          {result.segments.map((segment) => (
            <li key={segment.id}>
              <button
                type="button"
                onClick={() => seek(segment.start)}
                className="flex w-full items-baseline gap-3 px-3 py-2 text-start hover:bg-muted/60"
              >
                <span className="shrink-0 font-mono text-xs tabular-nums text-primary">
                  {formatClock(segment.start)}
                </span>
                <span dir="auto" className="flex-1 font-fa leading-relaxed">
                  {segment.text}
                </span>
              </button>
            </li>
          ))}
        </ol>
      )}

      {result.words && result.words.length > 0 && (
        <details className="mt-3 text-sm">
          <summary className="cursor-pointer text-muted-foreground">
            Word timestamps ({result.words.length})
          </summary>
          <p dir="auto" className="mt-2 flex flex-wrap gap-1.5 font-fa">
            {result.words.map((word, i) => (
              <button
                key={i}
                type="button"
                onClick={() => seek(word.start)}
                title={`${word.start.toFixed(2)}s – ${word.end.toFixed(2)}s`}
                className="rounded-md bg-muted px-1.5 py-0.5 hover:bg-primary/15"
              >
                {word.word}
              </button>
            ))}
          </p>
        </details>
      )}
    </article>
  );
}
