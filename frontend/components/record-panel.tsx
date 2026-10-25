"use client";

import { Mic, Square, Loader2 } from "lucide-react";
import { extensionFor, useRecorder } from "@/hooks/use-recorder";
import { cn, formatClock } from "@/lib/utils";

const BARS = 24;

export function RecordPanel({
  disabled,
  onRecorded,
}: {
  disabled: boolean;
  onRecorded: (audio: Blob, filename: string) => void;
}) {
  const { state, elapsed, level, error, start, stop } = useRecorder();
  const recording = state === "recording";

  const toggle = async () => {
    if (recording) {
      const blob = await stop();
      if (blob) onRecorded(blob, `recording.${extensionFor(blob.type)}`);
    } else {
      await start();
    }
  };

  return (
    <div className="flex flex-col items-center gap-5 py-6">
      <button
        type="button"
        onClick={toggle}
        disabled={(disabled && !recording) || state === "requesting"}
        aria-label={recording ? "Stop recording" : "Start recording"}
        className={cn(
          "flex size-24 items-center justify-center rounded-full text-white shadow-lg transition",
          "focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-primary/40",
          "disabled:cursor-not-allowed disabled:opacity-50",
          recording
            ? "animate-pulsering bg-destructive hover:bg-destructive/90"
            : "bg-primary hover:bg-primary/90",
        )}
      >
        {state === "requesting" ? (
          <Loader2 className="size-9 animate-spin" />
        ) : recording ? (
          <Square className="size-8 fill-current" />
        ) : (
          <Mic className="size-10" />
        )}
      </button>

      <div className="flex h-8 items-center gap-[3px]" aria-hidden>
        {Array.from({ length: BARS }, (_, i) => {
          // A symmetric envelope so the meter peaks in the middle.
          const shape = 1 - Math.abs(i - (BARS - 1) / 2) / (BARS / 2);
          const height = recording ? Math.max(0.08, level * (0.35 + 0.65 * shape)) : 0.08;
          return (
            <span
              key={i}
              className={cn(
                "w-1 rounded-full transition-[height] duration-75",
                recording ? "bg-destructive/80" : "bg-border",
              )}
              style={{ height: `${height * 100}%` }}
            />
          );
        })}
      </div>

      <p className="font-mono text-sm tabular-nums text-muted-foreground">
        {recording ? formatClock(elapsed) : "Click the microphone and speak"}
      </p>
      {error && <p className="max-w-sm text-center text-sm text-destructive">{error}</p>}
    </div>
  );
}
