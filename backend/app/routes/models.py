from fastapi import APIRouter, Depends, Query, Request

from app.asr.registry import ModelRegistry, ModelSpec
from app.errors import APIError
from app.schemas import Model, ModelEnvelope, ModelLinks, ModelList

router = APIRouter(tags=["models"])


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def to_model(spec: ModelSpec, registry: ModelRegistry) -> Model:
    return Model(
        id=spec.id,
        canonical_slug=spec.id,
        name=spec.name,
        created=spec.created,
        description=spec.description,
        hugging_face_id=spec.hugging_face_id,
        links=ModelLinks(details=f"/api/v1/models/{spec.id}/endpoints"),
        supported_parameters=spec.supported_parameters,
        languages=list(spec.languages),
        loaded=registry.is_loaded(spec.id),
        available=spec.id not in registry.unavailable,
        unavailable_reason=registry.unavailable.get(spec.id),
        max_chunk_seconds=spec.max_chunk_seconds,
    )


@router.get("/models", response_model=ModelList)
def list_models(
    registry: ModelRegistry = Depends(get_registry),
    output_modalities: str | None = Query(default=None),
    input_modalities: str | None = Query(default=None),
) -> ModelList:
    """Every model this server offers, in OpenRouter's catalog shape.

    The modality filters behave as on OpenRouter, so a client asking for
    ``output_modalities=transcription`` (or ``all``) gets the list and a client
    asking for text models gets an empty one.
    """
    if output_modalities and not _matches(output_modalities, "transcription"):
        return ModelList(data=[], total_count=0)
    if input_modalities and not _matches(input_modalities, "audio"):
        return ModelList(data=[], total_count=0)
    models = [to_model(spec, registry) for spec in registry.specs.values()]
    return ModelList(data=models, total_count=len(models))


@router.get("/models/{author}/{slug}", response_model=ModelEnvelope)
def get_model(
    author: str, slug: str, registry: ModelRegistry = Depends(get_registry)
) -> ModelEnvelope:
    try:
        spec = registry.resolve(f"{author}/{slug}")
    except APIError as exc:
        raise APIError(404, f"Model '{author}/{slug}' not found.") from exc
    return ModelEnvelope(data=to_model(spec, registry))


def _matches(query: str, modality: str) -> bool:
    wanted = {part.strip().lower() for part in query.split(",")}
    return "all" in wanted or modality in wanted
