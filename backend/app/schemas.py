"""Request and response bodies, shaped after OpenRouter's API reference.

- Models: https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties
- STT:    https://openrouter.ai/docs/api/api-reference/stt/create-transcription
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ── Models ──────────────────────────────────────────────────────────────────


class Architecture(BaseModel):
    modality: str = "audio->transcription"
    input_modalities: list[str] = ["audio"]
    output_modalities: list[str] = ["transcription"]
    instruct_type: str | None = None
    tokenizer: str = "Other"


class Pricing(BaseModel):
    """USD per unit, as decimal strings. Self-hosted, so everything is free."""

    prompt: str = "0"
    completion: str = "0"
    request: str = "0"
    audio: str = "0"


class TopProvider(BaseModel):
    is_moderated: bool = False
    context_length: int | None = None
    max_completion_tokens: int | None = None


class ModelLinks(BaseModel):
    details: str


class Model(BaseModel):
    id: str
    canonical_slug: str
    name: str
    created: int
    description: str = ""
    context_length: int | None = None
    hugging_face_id: str | None = None
    architecture: Architecture = Architecture()
    pricing: Pricing = Pricing()
    top_provider: TopProvider = TopProvider()
    per_request_limits: None = None
    supported_parameters: list[str] = ["language", "response_format", "timestamp_granularities"]
    default_parameters: None = None
    supported_voices: None = None
    links: ModelLinks
    # Not in OpenRouter's schema; useful to clients choosing a model.
    languages: list[str] = []
    loaded: bool = False
    #: False when this server cannot fetch the checkpoint (e.g. a private
    #: Hugging Face repo and no HF_TOKEN); ``unavailable_reason`` says why.
    available: bool = True
    unavailable_reason: str | None = None


class ListLinks(BaseModel):
    next: str | None = None


class ModelList(BaseModel):
    data: list[Model]
    total_count: int
    links: ListLinks = ListLinks()


class ModelEnvelope(BaseModel):
    data: Model


# ── Speech to text ──────────────────────────────────────────────────────────

ResponseFormat = Literal["json", "verbose_json"]
Granularity = Literal["segment", "word"]


class InputAudioInline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: str = Field(description="Raw base64 audio, without a data: URI prefix.")
    format: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9+._-]{0,15}$")


class InputAudioURL(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    format: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9+._-]{0,15}$")


class STTOptions(BaseModel):
    """Everything in a transcription request except the audio itself.

    OpenRouter-only routing fields (``provider``, ``trace``, ``session_id``,
    ``user``) are accepted and ignored, so a request built for OpenRouter works
    here unchanged. So are ``temperature`` and ``prompt``, which these models
    have no use for.
    """

    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    language: str | None = None
    temperature: float | None = None
    prompt: str | None = None
    response_format: ResponseFormat = "json"
    timestamp_granularities: list[Granularity] | None = None
    diarize: bool | None = None
    keyterms: list[str] | None = None


class STTRequest(STTOptions):
    """JSON body of POST /api/v1/audio/transcriptions."""

    input_audio: InputAudioInline | InputAudioURL


class Usage(BaseModel):
    seconds: float
    cost: float = 0.0


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class Word(BaseModel):
    word: str
    start: float
    end: float


class STTResponse(BaseModel):
    text: str
    usage: Usage
    # verbose_json only
    task: str | None = None
    language: str | None = None
    duration: float | None = None
    segments: list[Segment] | None = None
    words: list[Word] | None = None
