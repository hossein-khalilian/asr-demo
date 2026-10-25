"""The models this server offers, which of them are in memory, and the queue
in front of them.

All inference and model loading runs under one lock, one request at a time.
NeMo models are not safe to call concurrently, and serialising loads with
inference means a model is never evicted while it is transcribing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from starlette.concurrency import run_in_threadpool

from app.asr.engine import Engine, Transcript
from app.errors import APIError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    #: OpenRouter-style slug, ``author/model``, lowercase.
    id: str
    #: Hugging Face / NGC model name, or a path to a local ``.nemo`` file.
    source: str
    name: str
    description: str = ""
    #: "rnnt" or "ctc" for hybrid checkpoints; None keeps the saved default.
    decoder: str | None = None
    languages: tuple[str, ...] = ()
    created: int = 0
    aliases: tuple[str, ...] = field(default=())

    @property
    def hugging_face_id(self) -> str | None:
        return None if self.source.endswith(".nemo") else self.source


def load_specs(path: Path) -> list[ModelSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    specs: list[ModelSpec] = []
    for entry in raw["models"]:
        source = entry["source"]
        model_id = entry.get("id") or source.lower()
        specs.append(
            ModelSpec(
                id=model_id,
                source=source,
                name=entry.get("name") or model_id,
                description=entry.get("description", ""),
                decoder=entry.get("decoder"),
                languages=tuple(entry.get("languages", ())),
                created=int(entry.get("created", 0)),
                aliases=tuple(entry.get("aliases", ())),
            )
        )
    if not specs:
        raise ValueError(f"{path} lists no models")
    return specs


Loader = Callable[[ModelSpec], Engine]


class ModelRegistry:
    def __init__(
        self,
        specs: list[ModelSpec],
        *,
        loader: Loader,
        default_model: str = "",
        max_loaded: int = 2,
        max_queue: int = 16,
    ) -> None:
        self.specs = {spec.id: spec for spec in specs}
        # Clients may name a model by its id, its source (the HF repo, any
        # case) or a configured alias.
        self._names: dict[str, str] = {}
        for spec in specs:
            for name in (spec.id, spec.source, *spec.aliases):
                self._names[name.lower()] = spec.id
        self.default = self.resolve(default_model) if default_model else specs[0]
        self._loader = loader
        self._max_loaded = max_loaded
        self._max_queue = max_queue
        self._loaded: OrderedDict[str, Engine] = OrderedDict()
        self._lock = asyncio.Lock()
        self._waiting = 0
        self.busy = False

    # ── Lookup ───────────────────────────────────────────────────────────────

    def resolve(self, name: str | None) -> ModelSpec:
        if not name or not name.strip():
            return self.default
        model_id = self._names.get(name.strip().lower())
        if model_id is None:
            raise APIError(
                400,
                f"'{name}' is not a valid model ID.",
                metadata={"available": sorted(self.specs)},
            )
        return self.specs[model_id]

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded

    @property
    def loaded_ids(self) -> list[str]:
        return list(self._loaded)

    @property
    def queue_length(self) -> int:
        return self._waiting

    # ── Inference ────────────────────────────────────────────────────────────

    async def transcribe(self, spec: ModelSpec, wav_path: str, *, timestamps: bool) -> Transcript:
        if self._lock.locked() and self._waiting >= self._max_queue:
            raise APIError(
                503,
                "The server is busy; try again shortly.",
                metadata={"queue_length": self._waiting},
                headers={"Retry-After": "5"},
            )
        self._waiting += 1
        try:
            await self._lock.acquire()
        finally:
            self._waiting -= 1
        self.busy = True
        try:
            engine = await run_in_threadpool(self._get_or_load, spec)
            try:
                return await run_in_threadpool(engine.transcribe, wav_path, timestamps=timestamps)
            except Exception as exc:
                logger.exception("Transcription failed on %s", spec.id)
                raise APIError(500, f"Transcription failed: {exc}") from exc
        finally:
            self.busy = False
            self._lock.release()

    async def preload(self, model_ids: list[str]) -> None:
        for model_id in model_ids:
            spec = self.resolve(model_id)
            async with self._lock:
                try:
                    await run_in_threadpool(self._get_or_load, spec)
                except APIError as exc:
                    logger.error("Preloading %s failed: %s", spec.id, exc.message)

    def _get_or_load(self, spec: ModelSpec) -> Engine:
        """Caller holds the lock."""
        engine = self._loaded.get(spec.id)
        if engine is not None:
            self._loaded.move_to_end(spec.id)
            return engine
        while len(self._loaded) >= self._max_loaded:
            evicted_id, evicted = self._loaded.popitem(last=False)
            logger.info("Unloading %s to make room for %s", evicted_id, spec.id)
            evicted.unload()
        try:
            engine = self._loader(spec)
        except Exception as exc:
            logger.exception("Loading %s failed", spec.id)
            raise APIError(
                503, f"Model '{spec.id}' failed to load: {exc}", headers={"Retry-After": "30"}
            ) from exc
        self._loaded[spec.id] = engine
        return engine

    def preload_ids(self, setting: str) -> list[str]:
        setting = setting.strip().lower()
        if setting in ("", "none"):
            return []
        if setting == "default":
            return [self.default.id]
        if setting == "all":
            return list(self.specs)[: self._max_loaded]
        return [self.resolve(name).id for name in setting.split(",") if name.strip()]
