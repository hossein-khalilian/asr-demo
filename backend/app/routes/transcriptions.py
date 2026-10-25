"""POST /api/v1/audio/transcriptions — OpenRouter's speech-to-text endpoint.

Takes either body OpenRouter takes:

- ``application/json`` with ``input_audio: {data: <base64>, format}``
- ``multipart/form-data`` with a ``file`` part, OpenAI-style, which is what the
  OpenAI SDK's ``client.audio.transcriptions.create`` sends.
"""

import base64
import binascii
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from app.asr.chunking import merge, split_wav
from app.asr.engine import DecodeOptions, segments_from_words
from app.asr.registry import ModelRegistry, ModelSpec
from app.audio import decode_to_wav, filter_chain, format_from_filename, temp_dir
from app.errors import APIError
from app.routes.models import get_registry
from app.schemas import (
    PROVIDER_SLUG,
    AsrOptions,
    InputAudioURL,
    Segment,
    STTOptions,
    STTRequest,
    STTResponse,
    Usage,
    Word,
)
from app.settings import Settings, get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["speech-to-text"])

_LIST_FIELDS = ("timestamp_granularities", "keyterms")
_JSON_FIELDS = ("provider", "trace")
#: Overlap used when the server chunks audio the request did not ask to chunk.
DEFAULT_OVERLAP_SECONDS = 2.0


@router.post(
    "/audio/transcriptions",
    response_model=STTResponse,
    response_model_exclude_none=True,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {"schema": STTRequest.model_json_schema()},
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {
                            "file": {"type": "string", "format": "binary"},
                            "model": {"type": "string"},
                            "language": {"type": "string"},
                            "response_format": {
                                "type": "string",
                                "enum": ["json", "verbose_json"],
                            },
                            "timestamp_granularities[]": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["segment", "word"]},
                            },
                        },
                    }
                },
            },
            "required": True,
        }
    },
)
async def create_transcription(
    request: Request,
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    started = time.perf_counter()
    content_type = request.headers.get("content-type", "").lower()
    declared = int(request.headers.get("content-length") or 0)
    # Base64 inflates by 4/3; allow that plus room for the other fields.
    if declared > settings.max_upload_bytes * 4 // 3 + 64 * 1024:
        raise _too_large(settings)

    if content_type.startswith("multipart/form-data"):
        options, audio, fmt = await _read_multipart(request, settings)
    elif content_type.startswith("application/json"):
        options, audio, fmt = await _read_json(request, settings)
    else:
        raise APIError(400, "Send application/json or multipart/form-data.")

    spec = registry.resolve(options.model)
    tuning = _tuning_options(options)
    _check_options(options, tuning, spec)
    verbose = options.response_format == "verbose_json"
    decode = DecodeOptions(
        decoder=tuning.decoder,
        beam_size=tuning.beam_size,
        temperature=options.temperature if spec.framework == "transformers" else None,
    )

    with temp_dir() as workdir:
        filters = filter_chain(
            highpass=tuning.highpass, denoise=tuning.denoise, normalize=tuning.normalize
        )
        decoded = await decode_to_wav(audio, fmt, workdir, filters)
        if settings.max_audio_seconds and decoded.seconds > settings.max_audio_seconds:
            raise APIError(
                400,
                f"Audio is {decoded.seconds:.0f}s; the limit is {settings.max_audio_seconds:.0f}s. "
                "Split it into shorter chunks.",
            )
        chunk_seconds, overlap_seconds = _chunking(tuning, spec)
        chunks = split_wav(decoded.path, chunk_seconds, overlap_seconds, workdir)
        transcripts, timings = await registry.transcribe(
            spec,
            [str(chunk.path) for chunk in chunks],
            # Stitching overlapping chunks needs word times even for plain json.
            timestamps=verbose or len(chunks) > 1,
            decode=decode,
        )
        transcript = merge(chunks, transcripts, overlap_seconds)

    body = STTResponse(text=transcript.text, usage=Usage(seconds=decoded.seconds))
    if verbose:
        segments = transcript.segments
        if len(segments) <= 1 and transcript.words:
            # NeMo cuts segments at punctuation; models that emit none return
            # the whole clip as one segment. Cut at pauses instead.
            segments = segments_from_words(transcript.words)
        segments = segments or [(0.0, decoded.seconds, transcript.text)]
        body.task = "transcribe"
        body.language = spec.languages[0] if spec.languages else options.language
        body.duration = decoded.seconds
        body.segments = [
            Segment(id=i, start=start, end=end, text=text)
            for i, (start, end, text) in enumerate(segments)
        ]
        if "word" in (options.timestamp_granularities or []):
            body.words = [Word(word=w, start=s, end=e) for s, e, w in transcript.words]

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    # Total time includes waiting behind other requests; inference alone is
    # what says how fast the model is (RTF < 1 is faster than real time).
    logger.info(
        "Transcribed %.1fs of audio with %s%s: inference %dms (RTF %.2f), "
        "queued %dms, model load %dms, total %dms",
        decoded.seconds,
        spec.id,
        _describe(tuning, decode, len(chunks), chunk_seconds, overlap_seconds),
        timings.inference_ms,
        timings.inference_ms / 1000 / decoded.seconds,
        timings.queued_ms,
        timings.load_ms,
        elapsed_ms,
    )
    return JSONResponse(
        body.model_dump(exclude_none=True),
        headers={
            "X-Generation-Id": f"gen-{uuid.uuid4().hex}",
            "X-Model": spec.id,
            "X-Processing-Ms": str(elapsed_ms),
            "X-Queue-Ms": str(timings.queued_ms),
            "X-Load-Ms": str(timings.load_ms),
            "X-Inference-Ms": str(timings.inference_ms),
            "X-Chunks": str(len(chunks)),
        },
    )


async def _read_json(request: Request, settings: Settings) -> tuple[STTOptions, bytes, str]:
    raw = await request.body()
    try:
        req = STTRequest.model_validate_json(raw)
    except ValidationError as exc:
        raise _invalid(exc) from exc
    if isinstance(req.input_audio, InputAudioURL):
        raise APIError(400, "input_audio.url is not supported by this provider; send data.")
    try:
        audio = base64.b64decode(req.input_audio.data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise APIError(400, "input_audio.data is not valid base64.") from exc
    if len(audio) > settings.max_upload_bytes:
        raise _too_large(settings)
    return req, audio, req.input_audio.format


async def _read_multipart(
    request: Request, settings: Settings
) -> tuple[STTOptions, bytes, str | None]:
    form = await request.form(max_files=1, max_fields=64)
    try:
        if form.get("source_url"):
            raise APIError(400, "source_url is not supported by this provider; upload a file.")
        if "input_audio" in form:
            raise APIError(400, "Multipart requests take a file part, not input_audio.")
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise APIError(400, "file: a file part is required.")

        fields: dict[str, Any] = {}
        for key, value in form.multi_items():
            if isinstance(value, UploadFile):
                continue
            name = key.removesuffix("[]")
            if name in _LIST_FIELDS:
                fields.setdefault(name, []).append(value)
            elif name == "diarize":
                fields[name] = value.lower() in ("1", "true", "yes")
            elif name in _JSON_FIELDS:
                # OpenRouter: object fields travel JSON-encoded in multipart.
                try:
                    fields[name] = json.loads(value)
                except ValueError as exc:
                    raise APIError(400, f"{name}: must be a JSON-encoded object.") from exc
            else:
                fields[name] = value
        try:
            options = STTOptions.model_validate(fields)
        except ValidationError as exc:
            raise _invalid(exc) from exc

        audio = await upload.read(settings.max_upload_bytes + 1)
        if len(audio) > settings.max_upload_bytes:
            raise _too_large(settings)
        return options, audio, format_from_filename(upload.filename, upload.content_type)
    finally:
        await form.close()


def _tuning_options(options: STTOptions) -> AsrOptions:
    raw = ((options.provider and options.provider.options) or {}).get(PROVIDER_SLUG) or {}
    try:
        return AsrOptions.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"])
        prefix = f"provider.options.{PROVIDER_SLUG}"
        raise APIError(
            400, f"{prefix}.{where}: {first['msg']}" if where else f"{prefix}: {first['msg']}"
        ) from exc


def _check_options(options: STTOptions, tuning: AsrOptions, spec: ModelSpec) -> None:
    if tuning.decoder and "decoder" not in spec.supported_parameters:
        raise APIError(400, f"Model '{spec.id}' has no decoder choice; decoder is for NeMo models.")
    if options.diarize:
        raise APIError(400, f"Model '{spec.id}' does not support diarize.")
    if options.keyterms:
        raise APIError(400, f"Model '{spec.id}' does not support keyterms.")
    if options.language and spec.languages:
        # Accept regional spellings (fa-IR) of a supported language.
        base = options.language.strip().lower().replace("_", "-").split("-", 1)[0]
        if base not in spec.languages:
            raise APIError(
                400,
                f"Model '{spec.id}' transcribes {', '.join(spec.languages)} only, "
                f"not '{options.language}'.",
            )


def _chunking(tuning: AsrOptions, spec: ModelSpec) -> tuple[float, float]:
    """Chunk length and overlap for this request: what it asked for, kept
    within the model's window (Whisper's 30 s). A model with a window is
    always chunked, with a default overlap when the request set none."""
    chunk, overlap = tuning.chunk_seconds, tuning.overlap_seconds
    window = spec.max_chunk_seconds
    if window and (not chunk or chunk > window):
        if not chunk:
            overlap = DEFAULT_OVERLAP_SECONDS
        chunk = window
        overlap = min(overlap, chunk / 2 - 0.5)
    return chunk, overlap


def _describe(
    tuning: AsrOptions, decode: DecodeOptions, chunks: int, chunk: float, overlap: float
) -> str:
    """The non-default settings, for the log line ("" when untuned)."""
    parts = [
        f"{name}={value}"
        for name, value in tuning.model_dump(exclude_defaults=True).items()
        if name not in ("chunk_seconds", "overlap_seconds")
    ]
    if chunks > 1:
        parts.insert(0, f"{chunks} chunks of {chunk:g}s/{overlap:g}s")
    if decode.temperature:
        parts.append(f"temperature={decode.temperature:g}")
    return f" [{', '.join(parts)}]" if parts else ""


def _invalid(exc: ValidationError) -> APIError:
    first = exc.errors()[0]
    where = ".".join(str(part) for part in first["loc"])
    return APIError(400, f"{where}: {first['msg']}" if where else first["msg"])


def _too_large(settings: Settings) -> APIError:
    return APIError(413, f"Audio exceeds the {settings.max_upload_mb:g} MB limit.")
