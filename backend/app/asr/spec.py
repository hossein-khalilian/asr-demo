"""The model catalog: config/models.json parsed into :class:`ModelSpec`."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.settings import BACKEND_DIR

#: Model runtimes an entry can name in "framework".
FRAMEWORKS = ("nemo", "transformers")

#: Request parameters every model takes: OpenRouter's, then this server's
#: tuning options (``provider.options["asr-demo"]``, see schemas.AsrOptions).
_COMMON_PARAMETERS = (
    "language",
    "response_format",
    "timestamp_granularities",
    "chunk_seconds",
    "overlap_seconds",
    "beam_size",
    "normalize",
    "denoise",
    "highpass",
)
_FRAMEWORK_PARAMETERS = {
    "nemo": ("decoder",),
    "transformers": ("temperature",),
}


@dataclass(frozen=True)
class ModelSpec:
    #: OpenRouter-style slug, ``author/model``, lowercase.
    id: str
    #: Hugging Face / NGC model name, or a path to a local ``.nemo`` file.
    source: str
    name: str
    description: str = ""
    #: "nemo" (NeMo ASR checkpoints) or "transformers" (Whisper and other
    #: Hugging Face ``automatic-speech-recognition`` models).
    framework: str = "nemo"
    #: Directory or file, relative to backend/, to load from instead of
    #: downloading ``source``. The server never downloads such a model: until
    #: it exists, the model is listed as unavailable with the download command.
    local_path: str | None = None
    #: "rnnt" or "ctc" for hybrid NeMo checkpoints; None keeps the saved default.
    decoder: str | None = None
    #: Extra ``generate()`` arguments for transformers models, e.g.
    #: ``{"language": "fa", "task": "transcribe"}`` for Whisper.
    generate: tuple[tuple[str, Any], ...] = ()
    #: Longest audio the model takes in one pass. The server always chunks
    #: audio for such a model. Transformers models default to Whisper's 30 s.
    max_chunk_seconds: float | None = None
    languages: tuple[str, ...] = ()
    created: int = 0
    aliases: tuple[str, ...] = field(default=())

    @property
    def load_from(self) -> str:
        """What the engine loads: the local copy if configured, else ``source``."""
        if self.local_path:
            return str(BACKEND_DIR / self.local_path)
        return self.source

    @property
    def hugging_face_id(self) -> str | None:
        return None if self.source.endswith(".nemo") else self.source

    @property
    def supported_parameters(self) -> list[str]:
        """Reported in /models (OpenRouter's field) so clients show only what applies."""
        return [*_COMMON_PARAMETERS, *_FRAMEWORK_PARAMETERS[self.framework]]

    @property
    def download_hint(self) -> str:
        return f"hf download {self.source} --local-dir backend/{self.local_path}"


def load_specs(path: Path) -> list[ModelSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    specs: list[ModelSpec] = []
    for entry in raw["models"]:
        source = entry["source"]
        model_id = entry.get("id") or source.lower()
        framework = entry.get("framework", "nemo")
        if framework not in FRAMEWORKS:
            raise ValueError(
                f"{model_id}: framework must be one of {FRAMEWORKS}, not {framework!r}"
            )
        specs.append(
            ModelSpec(
                id=model_id,
                source=source,
                name=entry.get("name") or model_id,
                description=entry.get("description", ""),
                framework=framework,
                local_path=entry.get("local_path"),
                decoder=entry.get("decoder"),
                generate=tuple(sorted((entry.get("generate") or {}).items())),
                max_chunk_seconds=entry.get(
                    "max_chunk_seconds", 30 if framework == "transformers" else None
                ),
                languages=tuple(entry.get("languages", ())),
                created=int(entry.get("created", 0)),
                aliases=tuple(entry.get("aliases", ())),
            )
        )
    if not specs:
        raise ValueError(f"{path} lists no models")
    return specs
