"""The NeMo model wrapper. Everything here blocks; callers run it in a thread.

NeMo is imported lazily so the API, its tests and CI run without it installed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    #: (start, end, text) in seconds. Empty when the model gave no timestamps.
    segments: list[tuple[float, float, str]] = field(default_factory=list)
    words: list[tuple[float, float, str]] = field(default_factory=list)


class Engine(Protocol):
    def transcribe(self, wav_path: str, *, timestamps: bool) -> Transcript: ...

    def unload(self) -> None: ...


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class NemoEngine:
    """One loaded NeMo ASR checkpoint (CTC, RNNT or hybrid RNNT/CTC)."""

    def __init__(self, source: str, *, decoder: str | None, device: str) -> None:
        import torch
        from nemo.collections.asr.models import ASRModel

        self.source = source
        map_location = torch.device(device)
        logger.info("Loading %s on %s", source, device)
        if os.path.exists(source):
            model = ASRModel.restore_from(restore_path=source, map_location=map_location)
        else:
            model = ASRModel.from_pretrained(model_name=source, map_location=map_location)
        model.eval()

        if decoder:
            # Hybrid RNNT/CTC checkpoints carry both decoders; the saved config
            # picks one (usually RNNT). Only switch when the config asks to.
            try:
                model.change_decoding_strategy(decoder_type=decoder, verbose=False)
            except TypeError:
                logger.warning("%s is not a hybrid model; ignoring decoder=%s", source, decoder)
        self.model = model
        logger.info("Loaded %s", source)

    def transcribe(self, wav_path: str, *, timestamps: bool) -> Transcript:
        import torch

        with torch.inference_mode():
            hyp = self._run(wav_path, timestamps)
        text = (hyp if isinstance(hyp, str) else getattr(hyp, "text", "")) or ""
        transcript = Transcript(text=text.strip())
        if timestamps and not isinstance(hyp, str):
            stamps: dict[str, Any] = getattr(hyp, "timestamp", None) or {}
            transcript.segments = _spans(stamps.get("segment"), "segment")
            transcript.words = _spans(stamps.get("word"), "word")
        return transcript

    def _run(self, wav_path: str, timestamps: bool) -> Any:
        kwargs: dict[str, Any] = {"batch_size": 1, "verbose": False}
        if timestamps:
            kwargs["timestamps"] = True
        try:
            result = self.model.transcribe([wav_path], **kwargs)
        except TypeError:
            # Older NeMo: no `verbose` / `timestamps` keywords.
            result = self.model.transcribe([wav_path], batch_size=1)
        # Hybrid and RNNT models on older NeMo return (best, all_hypotheses).
        if isinstance(result, tuple):
            result = result[0]
        return result[0]

    def unload(self) -> None:
        del self.model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


_HF_TOKEN_HINT = "set HF_TOKEN in backend/.env to a token that can read it"


def access_error(exc: BaseException) -> str | None:
    """Why ``exc`` means the checkpoint cannot be fetched with the current
    credentials, or None if it is some other failure."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    if "GatedRepoError" in names:
        return f"gated on Hugging Face: accept its terms, then {_HF_TOKEN_HINT}"
    if "RepositoryNotFoundError" in names:
        return f"private or missing on Hugging Face: {_HF_TOKEN_HINT}"
    return None


def check_access(source: str) -> str | None:
    """None if ``source`` can be loaded, else why not.

    Asks the Hub for the repo's metadata (a few hundred bytes) with whatever
    token huggingface_hub finds (HF_TOKEN). Network trouble and a missing
    huggingface_hub count as accessible: only a definite "no" marks a model
    unavailable.
    """
    if os.path.exists(source):
        return None
    if source.endswith(".nemo"):
        return f"local checkpoint {source} does not exist"
    try:
        from huggingface_hub import model_info
    except ImportError:
        return None
    try:
        model_info(source)
    except Exception as exc:
        return access_error(exc)
    return None


def _spans(items: Any, key: str) -> list[tuple[float, float, str]]:
    spans: list[tuple[float, float, str]] = []
    for item in items or []:
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(item.get(key) or item.get("char") or "").strip()
        if text:
            spans.append((round(start, 3), round(end, 3), text))
    return spans


def segments_from_words(
    words: list[tuple[float, float, str]],
    *,
    min_pause: float = 0.6,
    long_pause: float = 1.5,
    min_seconds: float = 3.0,
    max_seconds: float = 12.0,
) -> list[tuple[float, float, str]]:
    """Group word spans into segments.

    A segment ends at a pause of ``long_pause`` or more; at a pause of
    ``min_pause`` once it is ``min_seconds`` long, so short breaths do not
    leave one-word segments; and before it would run past ``max_seconds``.
    """
    segments: list[tuple[float, float, str]] = []
    current: list[tuple[float, float, str]] = []

    def flush() -> None:
        segments.append((current[0][0], current[-1][1], " ".join(w[2] for w in current)))

    for word in words:
        if current:
            pause = word[0] - current[-1][1]
            length = current[-1][1] - current[0][0]
            if (
                pause >= long_pause
                or (pause >= min_pause and length >= min_seconds)
                or word[1] - current[0][0] > max_seconds
            ):
                flush()
                current = []
        current.append(word)
    if current:
        flush()
    return segments
