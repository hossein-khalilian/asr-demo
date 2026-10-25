"use client";

import { useEffect, useState } from "react";
import { fetchHealth } from "@/lib/api";
import type { Health } from "@/lib/types";
import { cn } from "@/lib/utils";

const POLL_MS = 10_000;

export function StatusPill() {
  const [health, setHealth] = useState<Health | null | undefined>(undefined);

  useEffect(() => {
    let active = true;
    const poll = async () => {
      try {
        const next = await fetchHealth();
        if (active) setHealth(next);
      } catch {
        if (active) setHealth(null);
      }
    };
    void poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  let label = "Connecting…";
  let tone = "bg-muted-foreground";
  if (health === null) {
    label = "Backend offline";
    tone = "bg-destructive";
  } else if (health) {
    if (health.busy) {
      label = health.queue_length > 0 ? `Busy · ${health.queue_length} queued` : "Transcribing";
      tone = "bg-amber-500";
    } else if (health.loaded_models.length === 0) {
      label = "Online · no model loaded yet";
      tone = "bg-amber-500";
    } else {
      label = "Online";
      tone = "bg-success";
    }
  }

  return (
    <span
      className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1 text-xs text-muted-foreground"
      title={health ? `Loaded: ${health.loaded_models.join(", ") || "none"}` : undefined}
    >
      <span className={cn("size-2 rounded-full", tone)} />
      {label}
    </span>
  );
}
