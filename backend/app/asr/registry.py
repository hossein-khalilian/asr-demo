"""The models this server offers, which of them are in memory, and the queue
in front of them.

All inference and model loading runs under one lock, one request at a time.
NeMo models are not safe to call concurrently, and serialising loads with
inference means a model is never evicted while it is transcribing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from starlette.concurrency import run_in_threadpool

from app.asr.engine import DecodeOptions, Engine, Transcript, UnsupportedOption, access_error
from app.asr.spec import ModelSpec, load_specs  # noqa: F401  (re-exported)
from app.errors import APIError

logger = logging.getLogger(__name__)


@dataclass
class Timings:
    """Where one request's time went, in milliseconds."""

    queued_ms: int = 0
    #: Loading the checkpoint; 0 when it was already in memory.
    load_ms: int = 0
    inference_ms: int = 0


Loader = Callable[[ModelSpec], Engine]
#: None if a model can be loaded, else the reason it cannot.
AccessChecker = Callable[[ModelSpec], str | None]


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
        #: Models that cannot be fetched with the current credentials, and why.
        #: Filled by :meth:`probe` at startup and by failed loads.
        self.unavailable: dict[str, str] = {}

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

    async def transcribe(
        self,
        spec: ModelSpec,
        wav_paths: list[str],
        *,
        timestamps: bool,
        decode: DecodeOptions = DecodeOptions(),
    ) -> tuple[list[Transcript], Timings]:
        """Transcribe every chunk of one request in a single turn of the queue."""
        timings = Timings()
        started = time.perf_counter()
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
            timings.queued_ms = _ms_since(started)
            mark = time.perf_counter()
            engine = await run_in_threadpool(self._get_or_load, spec)
            timings.load_ms = _ms_since(mark)
            mark = time.perf_counter()
            transcripts: list[Transcript] = []
            try:
                for path in wav_paths:
                    transcripts.append(
                        await run_in_threadpool(
                            engine.transcribe, path, timestamps=timestamps, decode=decode
                        )
                    )
            except UnsupportedOption as exc:
                raise APIError(400, f"Model '{spec.id}': {exc}.") from exc
            except Exception as exc:
                logger.exception("Transcription failed on %s", spec.id)
                raise APIError(500, f"Transcription failed: {exc}") from exc
            timings.inference_ms = _ms_since(mark)
            return transcripts, timings
        finally:
            self.busy = False
            self._lock.release()

    async def probe(self, checker: AccessChecker) -> None:
        """Find the models this server cannot fetch, so clients can see it in
        /models before asking for one."""
        for spec in self.specs.values():
            reason = await run_in_threadpool(checker, spec)
            if reason:
                logger.warning("Model %s is unavailable: %s", spec.id, reason)
                self.unavailable[spec.id] = reason
            else:
                self.unavailable.pop(spec.id, None)

    async def preload(self, model_ids: list[str]) -> None:
        for model_id in model_ids:
            spec = self.resolve(model_id)
            if spec.id in self.unavailable:
                logger.error("Not preloading %s: %s", spec.id, self.unavailable[spec.id])
                continue
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
        # Fail before evicting anything: a model that cannot be fetched must
        # not push out one that works.
        if spec.id in self.unavailable:
            raise self._unavailable_error(spec)
        while len(self._loaded) >= self._max_loaded:
            evicted_id, evicted = self._loaded.popitem(last=False)
            logger.info("Unloading %s to make room for %s", evicted_id, spec.id)
            evicted.unload()
        try:
            engine = self._loader(spec)
        except Exception as exc:
            reason = access_error(exc)
            if reason:
                logger.error("Model %s is unavailable: %s", spec.id, reason)
                self.unavailable[spec.id] = reason
                raise self._unavailable_error(spec) from exc
            logger.exception("Loading %s failed", spec.id)
            raise APIError(
                503, f"Model '{spec.id}' failed to load: {exc}", headers={"Retry-After": "30"}
            ) from exc
        self._loaded[spec.id] = engine
        return engine

    def _unavailable_error(self, spec: ModelSpec) -> APIError:
        # No Retry-After: this does not fix itself; it needs a config change.
        return APIError(
            503,
            f"Model '{spec.id}' is unavailable on this server: {self.unavailable[spec.id]}.",
            metadata={"source": spec.source},
        )

    def preload_ids(self, setting: str) -> list[str]:
        setting = setting.strip().lower()
        if setting in ("", "none"):
            return []
        if setting == "default":
            return [self.default.id]
        if setting == "all":
            return list(self.specs)[: self._max_loaded]
        return [self.resolve(name).id for name in setting.split(",") if name.strip()]


def _ms_since(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
