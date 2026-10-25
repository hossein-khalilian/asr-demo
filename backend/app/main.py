"""ASR demo API: self-hosted NeMo speech-to-text behind OpenRouter's API shape.

Point any OpenRouter or OpenAI client at ``http://<host>:8000/api/v1``:

    client = OpenAI(base_url="http://localhost:8000/api/v1", api_key="...")
    client.audio.transcriptions.create(model="hsekhalilian/speech-to-text-finetuning-01",
                                       file=open("sample.wav", "rb"))
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.asr.engine import NemoEngine, check_access, resolve_device
from app.asr.registry import AccessChecker, Loader, ModelRegistry, ModelSpec, load_specs
from app.auth import require_api_key
from app.errors import install_error_handlers
from app.routes import health, models, transcriptions
from app.settings import Settings, get_settings

logger = logging.getLogger("app")


def nemo_loader(settings: Settings) -> Loader:
    device = resolve_device(settings.asr_device)

    def load(spec: ModelSpec) -> NemoEngine:
        return NemoEngine(spec.source, decoder=spec.decoder, device=device)

    return load


def create_app(
    settings: Settings | None = None,
    loader: Loader | None = None,
    access_checker: AccessChecker = check_access,
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    registry = ModelRegistry(
        load_specs(settings.asr_models_file),
        loader=loader or nemo_loader(settings),
        default_model=settings.asr_default_model,
        max_loaded=settings.asr_max_loaded_models,
        max_queue=settings.max_queue,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not settings.api_keys:
            logger.warning("API_KEYS is empty: /api/v1 is open to anyone who can reach it.")

        # In the background so /health answers while a checkpoint downloads;
        # requests that arrive meanwhile queue behind the preload.
        async def startup() -> None:
            await registry.probe(access_checker)
            await registry.preload(registry.preload_ids(settings.asr_preload))

        task = asyncio.create_task(startup())
        yield
        task.cancel()

    app = FastAPI(
        title="ASR Demo API",
        version="2.0.0",
        description="Persian speech-to-text with NVIDIA NeMo, served in OpenRouter's API shape.",
        lifespan=lifespan,
    )
    app.state.registry = registry
    app.dependency_overrides[get_settings] = lambda: settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials="*" not in settings.cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Generation-Id", "X-Model", "X-Processing-Ms"],
    )
    install_error_handlers(app)

    api = [Depends(require_api_key)]
    app.include_router(models.router, prefix="/api/v1", dependencies=api)
    app.include_router(transcriptions.router, prefix="/api/v1", dependencies=api)
    app.include_router(health.router)
    return app


app = create_app()
