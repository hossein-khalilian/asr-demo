"use client";

import { useState } from "react";
import { ChevronDown, RotateCcw, SlidersHorizontal } from "lucide-react";
import { describeSettings, maxOverlap, type AsrSettings, type Decoder } from "@/lib/settings";
import { cn } from "@/lib/utils";

export function SettingsPanel({
  settings,
  supported,
  maxChunkSeconds,
  disabled,
  onChange,
  onReset,
}: {
  settings: AsrSettings;
  /** The selected model's `supported_parameters`; other controls are hidden. */
  supported: string[];
  /** The model's window (Whisper: 30 s): the server always chunks to it. */
  maxChunkSeconds: number | null;
  disabled: boolean;
  /** Called with only the fields that changed. */
  onChange: (patch: Partial<AsrSettings>) => void;
  onReset: () => void;
}) {
  const [open, setOpen] = useState(false);
  const has = (name: string) => supported.includes(name);
  const set = <K extends keyof AsrSettings>(key: K, value: AsrSettings[K]) =>
    onChange({ [key]: value } as Partial<AsrSettings>);
  const changed = describeSettings(settings, supported, maxChunkSeconds);
  const chunkMax = maxChunkSeconds ?? 120;
  const chunkSeconds = Math.min(settings.chunkSeconds, chunkMax);

  return (
    <div className="mt-4 rounded-xl border border-border">
      <div className="flex items-center gap-2 px-3 py-2.5">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="flex min-w-0 flex-1 items-center gap-2 text-start text-sm font-medium"
        >
          <SlidersHorizontal className="size-4 shrink-0 text-muted-foreground" />
          Tuning
          <span className="truncate text-xs font-normal text-muted-foreground">
            {changed.length ? changed.join(" · ") : "model defaults"}
          </span>
          <ChevronDown className={cn("ms-auto size-4 shrink-0 transition", open && "rotate-180")} />
        </button>
        {changed.length > 0 && (
          <button
            type="button"
            onClick={onReset}
            disabled={disabled}
            className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-50"
          >
            <RotateCcw className="size-3" />
            Reset
          </button>
        )}
      </div>

      {open && (
        <fieldset
          disabled={disabled}
          className="grid gap-5 border-t border-border px-3 py-4 disabled:opacity-60"
        >
          <Group title="Decoding">
            {has("decoder") && (
              <Field
                label="Decoder"
                hint="Hybrid models carry two decoders. CTC is often faster on CPU, and for some models more accurate. Compare on your audio."
              >
                <Segmented
                  value={settings.decoder}
                  options={[
                    ["default", "Model default"],
                    ["rnnt", "RNNT"],
                    ["ctc", "CTC"],
                  ]}
                  onChange={(v) => set("decoder", v as Decoder)}
                />
              </Field>
            )}
            {has("beam_size") && (
              <Slider
                label="Beam size"
                value={settings.beamSize}
                min={1}
                max={8}
                step={1}
                format={(v) => (v === 1 ? "1 (greedy)" : String(v))}
                hint={
                  has("decoder")
                    ? "Keeps several candidate transcripts. 2–3× slower on CPU, and NeMo gives no word timestamps with it."
                    : "Keeps several candidate transcripts. Slower; can fix rare words."
                }
                onChange={(v) => set("beamSize", v)}
              />
            )}
            {has("temperature") && (
              <Slider
                label="Temperature"
                value={settings.temperature}
                min={0}
                max={1}
                step={0.1}
                format={(v) => (v === 0 ? "0 (deterministic)" : v.toFixed(1))}
                hint="Above 0 samples instead of picking the likeliest token. Usually worse; useful to escape repetition loops."
                onChange={(v) => set("temperature", v)}
              />
            )}
          </Group>

          {has("chunk_seconds") && (
            <Group title="Long audio">
              <Toggle
                label="Split into chunks"
                hint={
                  maxChunkSeconds
                    ? `This model hears at most ${maxChunkSeconds} s at a time, so longer audio is always cut into ${maxChunkSeconds} s chunks with 2 s overlap. Turn this on to choose shorter chunks or another overlap.`
                    : "Transcribes overlapping pieces and stitches them by word timing. Bounds memory on long recordings and matches how abrisham-stt sends audio; usually a little slower."
                }
                checked={settings.chunking}
                onChange={(v) => set("chunking", v)}
              />
              {settings.chunking && (
                <>
                  <Slider
                    label="Chunk length"
                    value={chunkSeconds}
                    min={5}
                    max={chunkMax}
                    step={5}
                    format={(v) => `${v}s`}
                    onChange={(v) =>
                      onChange({
                        chunkSeconds: v,
                        overlapSeconds: Math.min(settings.overlapSeconds, maxOverlap(v)),
                      })
                    }
                  />
                  <Slider
                    label="Overlap"
                    value={Math.min(settings.overlapSeconds, maxOverlap(chunkSeconds))}
                    min={0}
                    max={maxOverlap(chunkSeconds)}
                    step={0.5}
                    format={(v) => `${v}s`}
                    hint="Audio each chunk shares with the next, so words at a cut are heard whole."
                    onChange={(v) => set("overlapSeconds", v)}
                  />
                </>
              )}
            </Group>
          )}

          {has("normalize") && (
            <Group title="Audio clean-up">
              <Toggle
                label="Normalize loudness"
                hint="Evens out quiet or uneven recordings."
                checked={settings.normalize}
                onChange={(v) => set("normalize", v)}
              />
              <Toggle
                label="Reduce noise"
                hint="Removes steady background noise (fans, hum). Can blur speech in clean audio."
                checked={settings.denoise}
                onChange={(v) => set("denoise", v)}
              />
              <Toggle
                label="High-pass filter"
                hint="Cuts rumble below 80 Hz (handling noise, traffic)."
                checked={settings.highpass}
                onChange={(v) => set("highpass", v)}
              />
            </Group>
          )}
        </fieldset>
      )}
    </div>
  );
}

function Group({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-3">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        {title}
      </h3>
      {children}
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="grid gap-1.5">
      <span className="text-sm">{label}</span>
      {children}
      {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  format,
  hint,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  hint?: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="grid gap-1.5">
      <span className="flex items-baseline justify-between text-sm">
        {label}
        <span className="font-mono text-xs tabular-nums text-muted-foreground">
          {format(value)}
        </span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-[oklch(var(--primary))]"
      />
      {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
    </label>
  );
}

function Toggle({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-start gap-2.5">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 size-4 shrink-0 accent-[oklch(var(--primary))]"
      />
      <span className="grid gap-0.5">
        <span className="text-sm">{label}</span>
        {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
      </span>
    </label>
  );
}

function Segmented({
  value,
  options,
  onChange,
}: {
  value: string;
  options: ReadonlyArray<readonly [string, string]>;
  onChange: (v: string) => void;
}) {
  return (
    <div className="grid grid-cols-3 gap-1 rounded-lg bg-muted p-1 text-xs font-medium">
      {options.map(([id, label]) => (
        <button
          key={id}
          type="button"
          aria-pressed={value === id}
          onClick={() => onChange(id)}
          className={cn(
            "rounded-md py-1.5 transition",
            value === id ? "bg-card shadow-sm" : "text-muted-foreground hover:text-foreground",
          )}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
