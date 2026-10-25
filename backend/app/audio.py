"""Turning whatever the client sent into the 16 kHz mono WAV NeMo expects.

Browsers record WebM/Opus and abrisham-stt sends MP3 chunks; NeMo's own loader
handles neither reliably, so every upload goes through ffmpeg first.
"""

import asyncio
import contextlib
import os
import re
import tempfile
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from app.errors import APIError

SAMPLE_RATE = 16_000
_SAFE_FORMAT = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9+._-]{0,15}$")


@dataclass(frozen=True)
class DecodedAudio:
    path: Path
    seconds: float


def format_from_filename(filename: str | None, content_type: str | None) -> str | None:
    """Audio format from the upload's extension, else its MIME subtype."""
    if filename:
        suffix = Path(filename).suffix.lstrip(".").lower()
        if suffix and _SAFE_FORMAT.match(suffix):
            return suffix
    if content_type and "/" in content_type:
        subtype = content_type.split("/", 1)[1].split(";", 1)[0].strip().lower()
        subtype = {"mpeg": "mp3", "x-wav": "wav", "wave": "wav", "x-m4a": "m4a"}.get(
            subtype, subtype
        )
        if _SAFE_FORMAT.match(subtype):
            return subtype
    return None


@contextlib.contextmanager
def temp_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="asr-") as path:
        yield Path(path)


async def decode_to_wav(data: bytes, fmt: str | None, workdir: Path) -> DecodedAudio:
    """Write ``data`` to ``workdir`` and convert it to 16 kHz mono PCM WAV.

    ``fmt == "pcm"`` means headerless signed 16-bit little-endian mono at
    16 kHz, as OpenRouter defines it; anything else is left to ffmpeg to sniff.
    """
    if not data:
        raise APIError(400, "Audio is empty.")
    fmt = (fmt or "").lower()
    if fmt and not _SAFE_FORMAT.match(fmt):
        raise APIError(400, f"Unsupported audio format '{fmt}'.")

    source = workdir / f"input.{fmt or 'bin'}"
    target = workdir / "audio.wav"
    source.write_bytes(data)

    input_args = ["-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1"] if fmt == "pcm" else []
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        *input_args,
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-y",
        str(target),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 or not target.exists():
        detail = stderr.decode(errors="replace").strip().splitlines()
        raise APIError(
            400,
            "Could not decode the audio. Send wav, mp3, flac, m4a, ogg, webm or aac.",
            metadata={"ffmpeg": detail[-1] if detail else None},
        )
    with contextlib.suppress(OSError):
        os.remove(source)

    with wave.open(str(target), "rb") as wav:
        seconds = wav.getnframes() / float(wav.getframerate() or SAMPLE_RATE)
    if seconds <= 0:
        raise APIError(400, "Audio contains no samples.")
    return DecodedAudio(path=target, seconds=round(seconds, 3))
