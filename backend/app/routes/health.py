from fastapi import APIRouter, Depends

from app.asr.registry import ModelRegistry
from app.routes.models import get_registry

router = APIRouter(tags=["health"])


@router.get("/health")
def health(registry: ModelRegistry = Depends(get_registry)) -> dict:
    """Liveness plus what the server is doing. Never blocks on the model."""
    return {
        "status": "ok",
        "default_model": registry.default.id,
        "loaded_models": registry.loaded_ids,
        "unavailable_models": registry.unavailable,
        "busy": registry.busy,
        "queue_length": registry.queue_length,
    }
