"""POST /api/v1/audio/transcriptions — OpenRouter's speech-to-text endpoint.

Takes either body OpenRouter takes:

- ``application/json`` with ``input_audio: {data: <base64>, format}``
- ``multipart/form-data`` with a ``file`` part, OpenAI-style, which is what the
  OpenAI SDK's ``client.audio.transcriptions.create`` sends.
"""

import base64
import binascii
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from app.asr.engine import segments_from_words
from app.asr.registry import ModelRegistry, ModelSpec
from app.audio import decode_to_wav, format_from_filename, temp_dir
from app.errors import APIError
from app.routes.models import get_registry
from app.schemas import (
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
    _check_options(options, spec)
    verbose = options.response_format == "verbose_json"

    with temp_dir() as workdir:
        decoded = await decode_to_wav(audio, fmt, workdir)
        if settings.max_audio_seconds and decoded.seconds > settings.max_audio_seconds:
            raise APIError(
                400,
                f"Audio is {decoded.seconds:.0f}s; the limit is {settings.max_audio_seconds:.0f}s. "
                "Split it into shorter chunks.",
            )
        transcript = await registry.transcribe(spec, str(decoded.path), timestamps=verbose)

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
    logger.info("Transcribed %.1fs of audio with %s in %dms", decoded.seconds, spec.id, elapsed_ms)
    return JSONResponse(
        body.model_dump(exclude_none=True),
        headers={
            "X-Generation-Id": f"gen-{uuid.uuid4().hex}",
            "X-Model": spec.id,
            "X-Processing-Ms": str(elapsed_ms),
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


def _check_options(options: STTOptions, spec: ModelSpec) -> None:
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


def _invalid(exc: ValidationError) -> APIError:
    first = exc.errors()[0]
    where = ".".join(str(part) for part in first["loc"])
    return APIError(400, f"{where}: {first['msg']}" if where else first["msg"])


def _too_large(settings: Settings) -> APIError:
    return APIError(413, f"Audio exceeds the {settings.max_upload_mb:g} MB limit.")
