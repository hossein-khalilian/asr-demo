# ASR Demo: Persian Speech-to-Text

Self-hosted Persian speech recognition with [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
FastConformer models, served through **OpenRouter's speech-to-text API**. Any OpenRouter or
OpenAI client can use it by changing the base URL. A Next.js app records or uploads audio and
shows the transcript with timestamps.

```
 Browser ──▶ Next.js (:3000) ──/api/v1/*──▶ FastAPI + NeMo (:8000)
             record / upload    same-origin   GET  /api/v1/models
             server-side key    proxy         POST /api/v1/audio/transcriptions
```

The API matches OpenRouter's, so abrisham-stt can add this server as a transcription provider
later. See [docs/abrisham-stt-integration.md](docs/abrisham-stt-integration.md).

---

## Quick start (Docker)

```bash
make dev        # copies .env.example → .env and backend/.env on first run
```

Open **http://localhost:3000**. The backend's Swagger UI is at http://localhost:8000/docs.

The first start downloads the default checkpoint (~450 MB) into the `model_cache` volume. `/health`
answers right away, and transcriptions wait in the queue until the model has loaded. Follow the
progress with `make dev-logs SERVICE=backend`.

> **Private checkpoints:** the `hsekhalilian/*` models are private on Hugging Face. Set
> `HF_TOKEN` in `backend/.env`, or set `ASR_DEFAULT_MODEL=nvidia/stt-fa-fastconformer-hybrid-large`
> to use NVIDIA's public model.

| Command | What it does |
|---|---|
| `make dev` / `make dev-down` | Dev stack with hot reload on both services |
| `make dev-logs SERVICE=backend` | Tail the logs of one service |
| `make prod` | Prod stack: baked images, served behind Traefik on `PUBLIC_URL` |
| `make deploy` | `git pull` + rebuild + restart the prod stack |
| `make … GPU=1` | Add `docker-compose.gpu.yml`: CUDA PyTorch and the host GPU (needs the NVIDIA Container Toolkit) |
| `make test` | Backend ruff + pytest and frontend lint + typecheck, without Docker or NeMo |

---

## API

The routes follow OpenRouter's API reference for [models](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties)
and [speech-to-text](https://openrouter.ai/docs/api/api-reference/stt/create-transcription).
When `API_KEYS` is set, every `/api/v1/*` call needs `Authorization: Bearer <key>`.

### `POST /api/v1/audio/transcriptions`

Multipart form (OpenAI-style, which is what the OpenAI SDK sends):

```bash
curl http://localhost:8000/api/v1/audio/transcriptions \
  -H "Authorization: Bearer $KEY" \
  -F file=@sample.mp3 \
  -F model=hsekhalilian/speech-to-text-finetuning-01
```

Or JSON with base64 audio:

```json
{
  "model": "hsekhalilian/speech-to-text-finetuning-01",
  "input_audio": { "data": "<base64>", "format": "mp3" },
  "response_format": "verbose_json",
  "timestamp_granularities": ["segment", "word"]
}
```

Response:

```json
{ "text": "…", "usage": { "seconds": 19.27, "cost": 0.0 } }
```

`verbose_json` adds `language`, `duration`, `segments`, and `words` when you request
`timestamp_granularities: ["word"]`. The response headers carry `X-Generation-Id` and `X-Model`,
plus timings in milliseconds: `X-Inference-Ms` (the model itself), `X-Queue-Ms` (waiting behind
other requests), `X-Load-Ms` (loading the checkpoint, 0 when it was already in memory) and
`X-Processing-Ms` (the whole request).

| Field | Notes |
|---|---|
| `model` | A model id from `/api/v1/models`, or its Hugging Face name. If omitted, the server default is used. |
| `file` / `input_audio` | Any format ffmpeg reads (wav, mp3, flac, m4a, ogg, webm, aac); `pcm` = raw s16le 16 kHz mono. 25 MB maximum. |
| `language` | Optional hint. A model accepts only the languages it lists (`fa`, `fa-IR`); anything else is a 400. |
| `response_format` | `json` (default) or `verbose_json`. `text`, `srt` and `vtt` return a 400, as on OpenRouter. |
| `prompt`, `temperature`, `provider`, `trace`, `session_id`, `user` | Accepted and ignored |
| `diarize`, `keyterms`, `input_audio.url`, `source_url` | Not supported: 400 |

Errors use OpenRouter's envelope, and the HTTP status equals `code`:

```json
{ "error": { "code": 400, "message": "'openai/whisper-1' is not a valid model ID.", "metadata": { "available": ["…"] } } }
```

### With the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/api/v1", api_key="sk-asr-…")
result = client.audio.transcriptions.create(
    model="hsekhalilian/speech-to-text-finetuning-01",
    file=open("sample.mp3", "rb"),
    language="fa",
)
print(result.text)
```

### Other routes

- `GET /api/v1/models`: the catalog in OpenRouter's shape (`architecture.output_modalities = ["transcription"]`, zero `pricing`), plus `languages`, `loaded`, and `available` / `unavailable_reason`. It honours `?output_modalities=`.
- `GET /api/v1/models/{author}/{slug}`: a single model.
- `GET /health`: no auth. Returns `default_model`, `loaded_models`, `unavailable_models`, `busy` and `queue_length`.

---

## Models

[`backend/config/models.json`](backend/config/models.json) lists the models:

```json
{
  "id": "hsekhalilian/speech-to-text-finetuning-01",
  "source": "hsekhalilian/Speech_To_Text_Finetuning_01",
  "name": "Persian FastConformer (fine-tune 01)",
  "decoder": "rnnt",
  "languages": ["fa"]
}
```

- `source` is a Hugging Face / NGC name, or a path to a local `.nemo` file. Put local files in
  `backend/models/`; the prod stack mounts it at `/app/models` (`"source": "models/my.nemo"`).
- `decoder` (optional) switches hybrid RNNT/CTC checkpoints to `rnnt` or `ctc`. Leave it out
  to keep the checkpoint's saved decoder.
- At startup the server asks the Hugging Face Hub whether it can fetch each model. A model it
  cannot fetch (a private repo and no `HF_TOKEN`) is listed with `available: false` and a reason,
  is greyed out in the UI, and answers 503 with that reason instead of a Hub traceback. Change
  `backend/.env` and recreate the backend to re-check.
- Models load on first use. At most `ASR_MAX_LOADED_MODELS` stay in memory; the least recently
  used one is unloaded to make room.
- One request runs at a time, because NeMo models are not safe to call concurrently. Up to
  `MAX_QUEUE` requests wait, and past that the server answers 503 with `Retry-After`.
- Models that emit no punctuation return the whole clip as one NeMo segment. `verbose_json` then
  builds segments from word timings, cutting at pauses.

---

## Configuration

**`backend/.env`** ([example](backend/.env.example)):

| Variable | Default | |
|---|---|---|
| `ASR_DEFAULT_MODEL` | first in models.json | Used when a request names no model |
| `ASR_PRELOAD` | `default` | `default`, `all`, `none`, or a comma list of ids |
| `ASR_MAX_LOADED_MODELS` | `2` | Models kept in memory |
| `ASR_DEVICE` | `auto` | `auto`, `cuda` or `cpu` |
| `ASR_NEMO_LOG_LEVEL` | `error` | NeMo's own logging. It warns on every request otherwise; use `warning` or `info` to debug a model |
| `API_KEYS` | empty | Comma-separated bearer keys. **Empty = open API**, for local use only |
| `MAX_UPLOAD_MB` / `MAX_AUDIO_SECONDS` / `MAX_QUEUE` | `25` / `600` / `16` | Request limits |
| `HF_TOKEN` | empty | Needed for the private `hsekhalilian/*` checkpoints |

**`.env`** at the repo root ([example](.env.example)) holds compose settings: ports, `PUBLIC_URL`/`BRANCH` for
Traefik, `ASR_API_KEY` (the key the frontend proxy sends; it must be one of `API_KEYS`), and the
base image registry and pip/PyTorch indexes.

**Frontend** (server-side only, never sent to the browser): `API_URL` and `ASR_API_KEY`.

---

## Development without Docker

```bash
# Backend (needs ffmpeg)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA index
pip install -r requirements-asr.txt -r requirements-dev.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
pnpm install
API_URL=http://localhost:8000 pnpm dev
```

The tests replace NeMo with a fake engine, so `pip install -r requirements-dev.txt` is enough to
run `pytest`. One test drives the real OpenAI SDK against the API.

---

## Project structure

```
asr-demo/
├── backend/
│   ├── app/
│   │   ├── main.py              # app factory: registry, CORS, auth, routers, background preload
│   │   ├── settings.py          # environment configuration
│   │   ├── schemas.py           # OpenRouter-shaped request/response models
│   │   ├── errors.py            # {"error": {code, message}} envelope
│   │   ├── auth.py              # bearer API keys
│   │   ├── audio.py             # ffmpeg → 16 kHz mono WAV
│   │   ├── asr/engine.py        # NeMo wrapper (lazy import), timestamps
│   │   ├── asr/registry.py      # model catalog, LRU loading, single-flight queue
│   │   └── routes/              # models, audio/transcriptions, health
│   ├── config/models.json
│   ├── tests/
│   ├── Dockerfile               # base → deps → dev | prod
│   └── requirements*.txt        # API / +NeMo / +GPU / +dev
├── frontend/                    # Next.js 16, React 19, Tailwind
│   ├── app/api/v1/[...path]/    # same-origin proxy to the backend (adds the API key)
│   ├── app/api/health/
│   ├── components/              # transcriber, record/upload panels, result card
│   ├── hooks/use-recorder.ts    # MediaRecorder + level meter, releases the mic on stop
│   └── Dockerfile               # dev | standalone prod
├── docker-compose.dev.yml       # hot reload, published ports
├── docker-compose.prod.yml      # baked images, Traefik on PUBLIC_URL
├── docker-compose.gpu.yml       # CUDA override for either stack
├── docs/abrisham-stt-integration.md
└── Makefile
```

## Logs

The backend writes one line per transcription and one per rejected request:

```
INFO app.routes.transcriptions: Transcribed 113.4s of audio with hsekhalilian/speech-to-text-finetuning-01: inference 26159ms (RTF 0.23), queued 0ms, model load 0ms, total 26380ms
WARNING app.errors: 400 POST /api/v1/audio/transcriptions: Audio is 660s; the limit is 600s. Split it into shorter chunks.
```

RTF (real-time factor) is inference time divided by audio length; below 1 is faster than real
time. Use `make dev-logs SERVICE=backend` to follow them.

## Notes

- **Microphone access** needs a secure context: `https://` or `localhost`.
- **The public page makes the API public.** The frontend proxy adds `ASR_API_KEY` to browser
  requests, so anyone who can open the page can transcribe. Callers that send their own
  `Authorization` header keep it.
- **Run one backend worker per process.** Each process holds its own model copy and queue. Scale
  out with more containers.
- **Long audio.** NeMo decodes a clip in one pass, so very long audio can exhaust GPU memory.
  `MAX_AUDIO_SECONDS` rejects anything over the limit; split long recordings into chunks, as
  abrisham-stt does (≤ 60 s).
