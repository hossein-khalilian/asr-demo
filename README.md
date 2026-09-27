# ASR Demo — Persian Speech-to-Text

A small web demo for **Automatic Speech Recognition (ASR)**. You record audio in the browser, the audio goes to a Python backend, and the backend transcribes it with an [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) **FastConformer Hybrid RNNT/CTC** model. The default models are Persian (Farsi) checkpoints fine-tuned from `nvidia/stt_fa_fastconformer_hybrid_large` and hosted on the Hugging Face Hub.

```
┌──────────────────────────┐   POST /transcribe (multipart, .webm)   ┌──────────────────────────────┐
│  Frontend (static HTML)  │ ──────────────────────────────────────▶ │  Backend (FastAPI + Uvicorn) │
│  MediaRecorder API       │                                         │  NeMo Hybrid RNNT/CTC model  │
│  http://localhost:8080   │ ◀────────────────────────────────────── │  http://localhost:8000       │
└──────────────────────────┘            { "text": "..." }            └──────────────────────────────┘
```

---

## Features

- One-click recording in the browser (🎤 Record / ⏹️ Stop) using the MediaRecorder API
- Transcription on the server with a NeMo `EncDecHybridRNNTCTCBPEModel`
- Loads a model from a **local `.nemo` file** or from a **Hugging Face / NGC model name**
- Inference runs in a thread-pool executor, so the async FastAPI event loop is not blocked
- A `/status` endpoint that reports whether a transcription is running
- No build step: plain HTML, CSS and JavaScript on the frontend

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| ASR model | [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) (`nemo_toolkit[asr]`) | FastConformer Hybrid RNNT/CTC BPE model |
| Model hosting | [Hugging Face Hub](https://huggingface.co/) | Pretrained and fine-tuned checkpoints |
| Backend framework | [FastAPI](https://fastapi.tiangolo.com/) | REST API |
| ASGI server | [Uvicorn](https://www.uvicorn.org/) | Serves the FastAPI app |
| File uploads | `python-multipart` | Parses `multipart/form-data` uploads |
| GPU | `cuda-python>=12.3` | CUDA bindings used by NeMo for GPU inference |
| Frontend | Vanilla HTML / CSS / JavaScript | Recording UI (`MediaRecorder`, `fetch`, `FormData`) |
| Static server | Python `http.server` | Serves the frontend during development |

---

## Project Structure

```
asr-demo/
├── backend/
│   ├── main.py            # FastAPI app: model selection, CORS, /transcribe and /status endpoints
│   ├── asr_model.py       # ASRModel wrapper: loads the NeMo model and runs async transcription
│   ├── requirements.txt   # Python dependencies
│   ├── run.sh             # Starts the API: uvicorn main:app --reload (port 8000)
│   └── .gitignore
└── frontend/
    ├── index.html         # UI: record/stop buttons, status line, transcription area
    ├── app.js             # Captures microphone audio and POSTs it to the backend
    ├── style.css          # Styling
    └── run.sh             # Serves the frontend: python -m http.server 8080
```

---

## How It Works

1. **Record.** When you click **Record**, `app.js` asks for microphone access with `navigator.mediaDevices.getUserMedia` and starts a `MediaRecorder`.
2. **Upload.** When you click **Stop**, the recorded chunks are combined into an `audio/webm` blob and sent as `file` (`recording.webm`) in a `multipart/form-data` POST to `http://localhost:8000/transcribe`.
3. **Save to a temporary file.** `main.py` writes the upload to a temporary file that keeps the original extension.
4. **Transcribe.** `ASRModel.transcribe()` calls `model.transcribe(path)` through `loop.run_in_executor`, so the blocking inference does not stall the server. It sets a `loading` flag while it runs.
5. **Respond.** The backend returns `{"text": "<transcription>"}` and deletes the temporary file. The frontend shows the text.

### Model loading (`asr_model.py`)

```python
if os.path.exists(model_name_or_path):
    model = EncDecHybridRNNTCTCBPEModel.restore_from(restore_path=model_name_or_path)   # local .nemo file
else:
    model = EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=model_name_or_path)  # HF Hub / NGC
```

The model loads **once, when the module is imported**, so the first startup can take a while (the checkpoint download is cached after that).

---

## Prerequisites

- **Python 3.10+** (check the NeMo release you install for its supported versions)
- **An NVIDIA GPU with CUDA 12.3+** is recommended. CPU inference may work but is much slower.
- **FFmpeg** is most likely needed. The browser sends WebM/Opus audio, and decoding it on the backend usually relies on `ffmpeg` being on your `PATH`.
- **A modern browser** that supports `MediaRecorder` (Chrome, Firefox, Edge).
- Build tools that NeMo may need (for example `build-essential`, `libsndfile1`, `Cython`), depending on your platform.

---

## Installation

```bash
git clone <repo-url> asr-demo
cd asr-demo/backend

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> Tip: install a PyTorch build that matches your CUDA version **before** `nemo_toolkit` (see [pytorch.org](https://pytorch.org/get-started/locally/)). Otherwise pip may install a mismatched CPU or CUDA build.

---

## Running

Open two terminals.

**1. Backend (port 8000)**

```bash
cd backend
./run.sh
```

This is the same as `uvicorn main:app --reload`. Wait until the log shows `Model has been loaded.`

**2. Frontend (port 8080)**

```bash
cd frontend
./run.sh
```

Then open **http://localhost:8080** in your browser, allow microphone access, click **Record**, speak, and click **Stop**.

> Browsers only allow microphone access in a secure context, meaning `https://` or `localhost`. If you open the frontend from another machine through a plain-HTTP IP address, recording will fail with `getUserMedia not supported`.

---

## API Reference

### `POST /transcribe`

Transcribes one uploaded audio file.

- **Content-Type:** `multipart/form-data`
- **Field:** `file`: the audio file (`.wav`, `.webm`, `.flac`, …)

```bash
curl -X POST -F "file=@sample.wav" http://localhost:8000/transcribe
```

**Response**

```json
{ "text": "transcribed text" }
```

### `GET /status`

Reports whether a transcription is running.

```bash
curl http://localhost:8000/status
```

```json
{ "loading": false }
```

FastAPI also generates interactive docs at **http://localhost:8000/docs** (Swagger UI) and **http://localhost:8000/redoc**.

---

## Configuration

Everything is set in `backend/main.py`.

### Choosing a model

`MODEL_NAME` is assigned several times, and **the last assignment wins**. Right now that is:

```python
MODEL_NAME = "hsekhalilian/Speech_To_Text_Finetuning_01"
```

Other options in the file:

| Model | Notes |
|-------|-------|
| `nvidia/stt_fa_fastconformer_hybrid_large` | Base NVIDIA Persian FastConformer hybrid model |
| `hsekhalilian/stt_fa_fastconformer_updated_tokenizer` | Fine-tuned with an updated tokenizer |
| `hsekhalilian/stt_fa_fastconformer_updated_tokenizer_01` | Updated-tokenizer variant |
| `hsekhalilian/Speech_To_Text_Finetuning_01` | Fine-tuned checkpoint (**current default**) |
| `hsekhalilian/Speech_To_Text_Finetuning_03_no_punc_with_encoder` | Fine-tuned without punctuation, encoder also trained |
| `hsekhalilian/speech-to-text-rnnt-finetuned` | RNNT-focused fine-tune |

You can also set `MODEL_NAME` to the path of a local `.nemo` file.

### Decoder type

```python
decoder_type = "rnnt"  # or "ctc"
```

Hybrid models have both an RNNT decoder and a CTC decoder. **Right now this setting has no effect**: the `change_decoding_strategy(decoder_type=...)` call in `asr_model.py` is commented out, so the model uses the decoder saved in its config (usually RNNT). To switch decoders, uncomment these lines in `ASRModel.__init__`:

```python
self.model.eval()
self.model.change_decoding_strategy(decoder_type=decoder_type)
```

### Backend URL and CORS

- The frontend's backend URL is hard-coded in `frontend/app.js` as `http://localhost:8000/transcribe`. Change it if the backend runs on a different host or port.
- CORS is fully open (`allow_origins=["*"]`). That is fine for a local demo; restrict it before any public deployment.

---

## Known Limitations and Notes

- **Development setup only.** `uvicorn --reload` and `python -m http.server` are not meant for production.
- **Single global model, one shared `loading` flag.** Concurrent requests share the model, and `/status` does not track overlapping requests correctly.
- **No streaming.** Audio is transcribed after recording stops, not in real time. The page title "ASR Streamer" does not mean it streams.
- **Unused UI hooks.** `index.html` includes a `#loading` indicator and `app.js` declares `statusInterval`, but neither is wired up. The frontend does not poll `/status` yet.
- **The microphone stays open.** The `MediaStream` tracks are not stopped after recording, so the browser's mic indicator may stay on until you close the tab.
- **Startup time.** The model loads when the module is imported. The first run downloads the checkpoint (hundreds of MB), and `--reload` reloads the model after every code change.

---

## Possible Improvements

- Read `MODEL_NAME`, `decoder_type` and the backend URL from environment variables or a config file
- Real-time streaming transcription over WebSockets with NeMo's cache-aware streaming
- A file-upload option alongside microphone recording
- Show the `#loading` indicator and poll `/status` from the frontend
- Package with Docker (a CUDA base image and FFmpeg) for reproducible deployment
- Stop the `MediaStream` tracks after each recording
