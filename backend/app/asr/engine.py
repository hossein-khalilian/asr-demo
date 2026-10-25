"""The NeMo model wrapper. Everything here blocks; callers run it in a thread.

NeMo is imported lazily so the API, its tests and CI run without it installed.
"""

from __future__ import annotations

import copy
import logging
import os
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from app.asr.spec import ModelSpec

logger = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    #: (start, end, text) in seconds. Empty when the model gave no timestamps.
    segments: list[tuple[float, float, str]] = field(default_factory=list)
    words: list[tuple[float, float, str]] = field(default_factory=list)


@dataclass(frozen=True)
class DecodeOptions:
    """Per-request decoding settings. The defaults change nothing."""

    #: Hybrid NeMo models: "rnnt" or "ctc"; None keeps the model's default.
    decoder: str | None = None
    #: 1 = greedy. NeMo cannot give word timestamps with beam search.
    beam_size: int = 1
    #: Whisper sampling temperature; None or 0 = deterministic.
    temperature: float | None = None


class UnsupportedOption(ValueError):
    """A decoding setting this model cannot honour (answered with a 400)."""


class Engine(Protocol):
    def transcribe(
        self, wav_path: str, *, timestamps: bool, decode: DecodeOptions = DecodeOptions()
    ) -> Transcript: ...

    def unload(self) -> None: ...


class _NemoLevelFilter(logging.Filter):
    """Drops records from NeMo's own code below ``level``.

    A filter rather than a level because NeMo's ``transcribe()`` forces its
    logger to WARNING for every call, which a plain level would not survive.
    Matching on the source path also catches NeMo modules that log through
    the root logger (Lhotse's "Initializing Lhotse CutSet" on every request).
    """

    def __init__(self, level: int) -> None:
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= self.level:
            return True
        return "/nemo/" not in record.pathname.replace("\\", "/")


_nemo_filter: _NemoLevelFilter | None = None


def quiet_nemo(level: str) -> None:
    """Hide NeMo's (and NVIDIA OneLogger's) records below ``level``."""
    global _nemo_filter
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        logger.warning("Unknown ASR_FRAMEWORK_LOG_LEVEL %r; leaving NeMo's logging as is", level)
        return
    logging.getLogger("nv_one_logger").setLevel(numeric)
    if _nemo_filter is None:
        _nemo_filter = _NemoLevelFilter(numeric)
        # NeMo's logger does not propagate, so it needs the filter itself.
        logging.getLogger("nemo_logger").addFilter(_nemo_filter)
        logging.getLogger().addFilter(_nemo_filter)
    _nemo_filter.level = numeric


def quiet_transformers(level: str) -> None:
    """Apply ``level`` to transformers' logger (deprecation and generation notices)."""
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        return
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity(numeric)


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

    def __init__(
        self, source: str, *, decoder: str | None, device: str, log_level: str = "error"
    ) -> None:
        # Before importing the ASR collection, which logs warnings on import.
        quiet_nemo(log_level)
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

        # Hybrid RNNT/CTC checkpoints carry both decoders; the saved config
        # picks one (usually RNNT).
        self.hybrid = hasattr(model, "cur_decoder")
        if decoder:
            if self.hybrid:
                model.change_decoding_strategy(decoder_type=decoder, verbose=False)
            else:
                logger.warning("%s is not a hybrid model; ignoring decoder=%s", source, decoder)
        self.model = model
        # What a request without settings gets, and the configs beam search
        # starts from. change_decoding_strategy() overwrites model.cfg, so
        # keep copies.
        self.default_decoder = model.cur_decoder if self.hybrid else self._single_decoder()
        self._base_cfg = {self.default_decoder: copy.deepcopy(model.cfg.decoding)}
        if self.hybrid:
            self._base_cfg = {
                "rnnt": copy.deepcopy(model.cfg.decoding),
                "ctc": copy.deepcopy(model.cfg.aux_ctc.decoding),
            }
        self._current = (self.default_decoder, 1)
        logger.info("Loaded %s", source)

    def _single_decoder(self) -> str:
        return "ctc" if "ctc" in type(self.model).__name__.lower() else "rnnt"

    def _configure(self, decode: DecodeOptions) -> None:
        """Switch decoder and greedy/beam search, only when they change."""
        from omegaconf import open_dict

        if decode.decoder and not self.hybrid and decode.decoder != self.default_decoder:
            raise UnsupportedOption(f"this model has only a {self.default_decoder.upper()} decoder")
        decoder = decode.decoder or self.default_decoder
        wanted = (decoder, decode.beam_size)
        if wanted == self._current:
            return
        cfg = copy.deepcopy(self._base_cfg[decoder])
        if decode.beam_size > 1:
            with open_dict(cfg):
                # The batched beam searches: pure PyTorch, no KenLM, and the
                # fastest NeMo offers (measured on CPU, 2026-10).
                cfg.strategy = "beam_batch" if decoder == "ctc" else "malsd_batch"
                cfg.beam.beam_size = decode.beam_size
        kwargs = {"decoder_type": decoder} if self.hybrid else {}
        self.model.change_decoding_strategy(decoding_cfg=cfg, verbose=False, **kwargs)
        self._current = wanted

    def transcribe(
        self, wav_path: str, *, timestamps: bool, decode: DecodeOptions = DecodeOptions()
    ) -> Transcript:
        import torch

        self._configure(decode)
        # NeMo refuses timestamps with beam search.
        timestamps = timestamps and decode.beam_size == 1
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
        hyp = result[0]
        # RNNT beam search returns an N-best list per file; the first is best.
        while isinstance(hyp, (list, tuple)):
            hyp = hyp[0]
        return hyp

    def unload(self) -> None:
        del self.model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class TransformersEngine:
    """A Hugging Face ``automatic-speech-recognition`` model, e.g. Whisper.

    Whisper reads one 30-second window at a time, and is never asked for
    timestamp tokens: fine-tunes trained without them (C1Tech's Persian
    Whisper, for one) emit an empty transcript when asked, and Whisper's own
    long-form decoding needs them. So the server cuts audio into windows of at
    most :attr:`WINDOW_SECONDS` before it gets here (``max_chunk_seconds``),
    and segments come from those chunks. Returns text only.
    """

    WINDOW_SECONDS = 30

    def __init__(
        self,
        path: str,
        *,
        device: str,
        generate_kwargs: dict[str, Any] | None = None,
        log_level: str = "error",
    ) -> None:
        quiet_transformers(log_level)
        import torch
        from transformers import pipeline

        logger.info("Loading %s on %s", path, device)
        self.generate_kwargs = generate_kwargs or {}
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=path,
            device=device,
            dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            # A local copy must be complete; never fall back to downloading.
            model_kwargs={"local_files_only": os.path.exists(path)},
        )
        self.is_whisper = self.pipe.type == "seq2seq_whisper"
        logger.info("Loaded %s", path)

    def transcribe(
        self, wav_path: str, *, timestamps: bool, decode: DecodeOptions = DecodeOptions()
    ) -> Transcript:
        import numpy as np

        if decode.decoder:
            raise UnsupportedOption("decoder applies to hybrid NeMo models only")
        generate_kwargs = dict(self.generate_kwargs)
        if decode.beam_size > 1:
            generate_kwargs["num_beams"] = decode.beam_size
        if decode.temperature:
            generate_kwargs.update(temperature=decode.temperature, do_sample=True)

        with wave.open(wav_path, "rb") as wav:
            rate = wav.getframerate()
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        duration = len(samples) / float(rate)
        audio = {"raw": samples.astype(np.float32) / 32768.0, "sampling_rate": rate}
        kwargs: dict[str, Any] = {}
        if self.is_whisper:
            kwargs["generate_kwargs"] = generate_kwargs
        if duration > self.WINDOW_SECONDS + 0.5:
            # The route keeps chunks within the window; should one ever be
            # longer, let the pipeline chunk it rather than silently truncate
            # it to the first 30 s.
            logger.warning("%.1fs chunk exceeds the %ss window", duration, self.WINDOW_SECONDS)
            kwargs.update(chunk_length_s=self.WINDOW_SECONDS, batch_size=1)
        out = self.pipe(audio, **kwargs)
        return Transcript(text=(out.get("text") or "").strip())

    def unload(self) -> None:
        del self.pipe
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


#: Files that hold a transformers model's weights; a local copy without one
#: was interrupted mid-download.
_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def check_access(spec: ModelSpec) -> str | None:
    """None if ``spec`` can be loaded, else why not.

    A model with ``local_path`` is checked on disk only: the server never
    downloads it. Otherwise this asks the Hub for the repo's metadata (a few
    hundred bytes) with whatever token huggingface_hub finds (HF_TOKEN).
    Network trouble and a missing huggingface_hub count as accessible: only a
    definite "no" marks a model unavailable.
    """
    if spec.local_path:
        path = Path(spec.load_from)
        if path.is_file():
            return None
        if not (path / "config.json").is_file():
            return f"not downloaded yet: run `{spec.download_hint}`"
        if not any((path / name).is_file() for name in _WEIGHT_FILES):
            return f"download incomplete (no model weights): rerun `{spec.download_hint}`"
        return None
    if os.path.exists(spec.source):
        return None
    if spec.source.endswith(".nemo"):
        return f"local checkpoint {spec.source} does not exist"
    try:
        from huggingface_hub import model_info
    except ImportError:
        return None
    try:
        model_info(spec.source)
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
