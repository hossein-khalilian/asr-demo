"""Request and response bodies, shaped after OpenRouter's API reference.

- Models: https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties
- STT:    https://openrouter.ai/docs/api/api-reference/stt/create-transcription
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    #: Set for models that take limited audio per pass (Whisper: 30 s); the
    #: server always chunks audio for them, to at most this length.
    max_chunk_seconds: float | None = None


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


#: The key this server reads in OpenRouter's ``provider.options``. OpenRouter
#: forwards to each provider only the options under its own slug, so tuning
#: settings for this server go there: ``{"provider": {"options": {"asr-demo": {...}}}}``.
PROVIDER_SLUG = "asr-demo"


class AsrOptions(BaseModel):
    """Inference settings, sent as ``provider.options["asr-demo"]``.

    Every default reproduces the untuned behaviour, so a request without
    options transcribes exactly as before.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_seconds: float = Field(
        default=0,
        ge=0,
        le=300,
        description="Split the audio into chunks of this many seconds (0 = one pass).",
    )
    overlap_seconds: float = Field(
        default=0, ge=0, le=10, description="Seconds each chunk shares with the next."
    )
    decoder: Literal["rnnt", "ctc"] | None = Field(
        default=None, description="Hybrid NeMo models: which decoder. None = the model's default."
    )
    beam_size: int = Field(
        default=1, ge=1, le=16, description="1 = greedy. Beam search has no word timestamps."
    )
    normalize: bool = Field(default=False, description="Normalise loudness (EBU R128).")
    denoise: bool = Field(default=False, description="FFT noise reduction.")
    highpass: bool = Field(default=False, description="Cut rumble below 80 Hz.")

    @model_validator(mode="after")
    def _check_chunking(self) -> "AsrOptions":
        if self.chunk_seconds and self.chunk_seconds < 5:
            raise ValueError("chunk_seconds must be 0 (off) or at least 5")
        if self.chunk_seconds and self.overlap_seconds * 2 >= self.chunk_seconds:
            raise ValueError("overlap_seconds must be less than half of chunk_seconds")
        return self


class ProviderPreferences(BaseModel):
    """OpenRouter's ``provider`` object. Only ``options`` means anything here."""

    model_config = ConfigDict(extra="ignore")

    options: dict[str, Any] | None = None


class STTOptions(BaseModel):
    """Everything in a transcription request except the audio itself.

    OpenRouter-only routing fields (``provider``, ``trace``, ``session_id``,
    ``user``) are accepted and ignored, so a request built for OpenRouter works
    here unchanged, and so is ``prompt``. ``temperature`` applies to Whisper
    models; NeMo models ignore it.
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
    provider: ProviderPreferences | None = None


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
