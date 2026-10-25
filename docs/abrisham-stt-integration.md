# Using asr-demo as an abrisham-stt provider

asr-demo serves OpenRouter's speech-to-text API, so abrisham-stt can call it the
way it already calls dedicated ASR models such as `qwen/qwen3-asr-flash` on
OpenRouter: `client.audio.transcriptions.create(model=…, file=…, language=…)`
with the OpenAI SDK pointed at a different `base_url`. Nothing on the asr-demo
side needs to change. This page lists the abrisham-stt changes for when we do it.

## What abrisham-stt already sends, and what asr-demo answers

| abrisham-stt does | asr-demo |
|---|---|
| `POST {base_url}/audio/transcriptions`, multipart, `file` = mono MP3 chunk | Decodes any ffmpeg format, so MP3 works |
| Chunks dedicated-ASR audio to ≤ 60 s (`DEDICATED_ASR_MAX_CHUNK_SECONDS`) | Default `MAX_AUDIO_SECONDS=600`, so there is room to spare |
| Sends `language` when the user picked one | Accepts `fa` / `fa-IR`; anything else is a 400, which abrisham-stt's failover treats as a provider error |
| Reads `.text` | `{"text": "…", "usage": {"seconds": 12.3, "cost": 0}}` |
| `Authorization: Bearer <key>` | Checked against `API_KEYS` |
| Failover on 5xx / 429 | Busy queue → 503 + `Retry-After`; model load failure → 503 |

`tests/test_api.py::test_openai_sdk_compat` runs that exact SDK call against the API.

## Changes in abrisham-stt

1. **Settings** (`backend/app/utils/settings.py`):

   ```python
   asrdemo_api_key: str = ""
   asrdemo_base_url: str = ""   # e.g. https://asr.example.com/api/v1
   ```

2. **Provider** (`backend/app/services/ai_providers.py`): subclass `AIProvider`
   and register it in `_PROVIDERS`.

   ```python
   class AsrDemoProvider(AIProvider):
       """Self-hosted NeMo models (github.com/hossein-khalilian/asr-demo)."""

       id = "asrdemo"
       name = "ASR Demo (NeMo)"
       api_key_setting = "asrdemo_api_key"
       default_models = {
           "transcription": "asrdemo:hsekhalilian/speech-to-text-finetuning-01",
       }

       @property
       def base_url(self) -> str:
           return settings.asrdemo_base_url.rstrip("/")
   ```

   `default_models` only has `transcription`. `ai_defaults` reads
   `provider.default_models[kind]` for every configured provider when it builds
   auto-mode fallbacks, so it must skip providers that do not serve a kind
   (`if kind in provider.default_models`). Otherwise auto mode for `nlp`,
   `embedding` and the other kinds raises `KeyError` once this provider is
   configured.

3. **Routing** (`backend/app/services/transcription.py`): add the model keys to
   `DEDICATED_TRANSCRIPTION_MODELS` so they go through `/audio/transcriptions`
   instead of chat-completions:

   ```python
   "asrdemo:hsekhalilian/speech-to-text-finetuning-01",
   "asrdemo:nvidia/stt-fa-fastconformer-hybrid-large",
   ```

   Model ids are listed at `GET {base_url}/models`.

4. **Prices**: the models are free per call. Either leave
   `fetch_price_catalog` returning `[]` and add zero-price rows, or implement it
   against `GET {base_url}/models` (each model has `pricing.prompt == "0"`).

5. **Frontend**: add the models to `TRANSCRIPTION_MODELS` in
   `frontend/features/transcribe/constants.ts`.

## Network

- **Public URL** (simplest): `ASRDEMO_BASE_URL=https://<PUBLIC_URL>/api/v1`. The
  asr-demo frontend proxies `/api/v1/*` to its backend and passes the caller's
  `Authorization` header through.
- **Same host, private network**: join both stacks to a shared external
  network, giving the asr-demo backend a unique alias (for example
  `asr-main-backend`). Do not rely on the service name `backend`: abrisham-stt's
  backend uses the same name, and Docker DNS would resolve both.

## Caveats

- **Throughput.** One request runs at a time per backend container, and the
  rest queue up to `MAX_QUEUE`. abrisham-stt's chunked jobs fire chunks in
  sequence per job, but several concurrent jobs will queue. Run more backend
  replicas (one GPU each) behind a load balancer if that matters.
- **The public demo page makes the API public.** The frontend proxy adds
  `ASR_API_KEY` to browser requests, so anyone who can open the page can
  transcribe. Put the frontend behind auth if that matters.
