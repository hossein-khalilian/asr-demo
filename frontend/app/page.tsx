import { AudioLines } from "lucide-react";
import { StatusPill } from "@/components/status-pill";
import { Transcriber } from "@/components/transcriber";

export default function Home() {
  return (
    <main className="mx-auto flex w-full max-w-2xl flex-col gap-8 px-4 py-10 sm:py-16">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className="flex size-10 items-center justify-center rounded-xl bg-primary text-primary-foreground">
            <AudioLines className="size-5" />
          </span>
          <div>
            <h1 className="text-xl font-semibold tracking-tight">ASR Demo</h1>
            <p className="text-sm text-muted-foreground">Persian speech to text with NVIDIA NeMo</p>
          </div>
        </div>
        <StatusPill />
      </header>
      <Transcriber />
      <footer className="text-center text-xs text-muted-foreground">
        Served through an OpenRouter-compatible API ·{" "}
        <code className="font-mono">POST /api/v1/audio/transcriptions</code>
      </footer>
    </main>
  );
}
