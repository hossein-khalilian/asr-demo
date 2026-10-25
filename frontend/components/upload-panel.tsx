"use client";

import { useEffect, useRef, useState } from "react";
import { FileAudio, Upload, X } from "lucide-react";
import { cn, formatBytes } from "@/lib/utils";

const MAX_BYTES = 25 * 1024 * 1024;

export function UploadPanel({
  disabled,
  onSubmit,
}: {
  disabled: boolean;
  onSubmit: (audio: Blob, filename: string) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [picked, setPicked] = useState<{ file: File; url: string } | null>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Revoke the preview URL whenever the pick changes or the panel unmounts.
  useEffect(() => (picked ? () => URL.revokeObjectURL(picked.url) : undefined), [picked]);

  const choose = (next: File | undefined) => {
    setError(null);
    if (!next) return;
    if (!next.type.startsWith("audio/") && !next.type.startsWith("video/") && next.type !== "") {
      setError("That does not look like an audio file.");
      return;
    }
    if (next.size > MAX_BYTES) {
      setError(`The file is ${formatBytes(next.size)}; the limit is 25 MB.`);
      return;
    }
    setPicked({ file: next, url: URL.createObjectURL(next) });
  };

  if (picked) {
    const { file, url } = picked;
    return (
      <div className="flex flex-col gap-4 py-4">
        <div className="flex items-center gap-3 rounded-xl border border-border bg-muted/50 p-3">
          <FileAudio className="size-8 shrink-0 text-primary" />
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">{file.name}</p>
            <p className="text-xs text-muted-foreground">{formatBytes(file.size)}</p>
          </div>
          <button
            type="button"
            onClick={() => setPicked(null)}
            className="rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground"
            aria-label="Remove file"
          >
            <X className="size-4" />
          </button>
        </div>
        <audio src={url} controls className="w-full" />
        <button
          type="button"
          disabled={disabled}
          onClick={() => onSubmit(file, file.name)}
          className="rounded-xl bg-primary px-4 py-2.5 font-medium text-primary-foreground transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Transcribe
        </button>
      </div>
    );
  }

  return (
    <div className="py-4">
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          choose(e.dataTransfer.files[0]);
        }}
        className={cn(
          "flex w-full flex-col items-center gap-3 rounded-2xl border-2 border-dashed px-6 py-10 text-center transition",
          dragging
            ? "border-primary bg-primary/5"
            : "border-border hover:border-primary/50 hover:bg-muted/50",
        )}
      >
        <Upload className="size-8 text-muted-foreground" />
        <span className="text-sm font-medium">Drop an audio file or click to choose</span>
        <span className="text-xs text-muted-foreground">
          wav, mp3, flac, m4a, ogg, webm · up to 25 MB
        </span>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,video/webm,video/mp4"
        className="hidden"
        onChange={(e) => {
          choose(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      {error && <p className="mt-3 text-center text-sm text-destructive">{error}</p>}
    </div>
  );
}
