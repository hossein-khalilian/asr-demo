"""Runtime configuration, read from the environment (and backend/.env)."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent

CommaList = Annotated[list[str], NoDecode]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Models ────────────────────────────────────────────────────────────────
    #: JSON file listing the models this server offers (see config/models.json).
    asr_models_file: Path = BACKEND_DIR / "config" / "models.json"
    #: Model used when a request names none. Empty → the first model in the file.
    asr_default_model: str = ""
    #: Models to load at startup: "default", "all", "none", or a comma list of ids.
    #: Anything not preloaded is loaded on first use.
    asr_preload: str = "default"
    #: How many models stay in memory at once; the least recently used is evicted.
    asr_max_loaded_models: int = Field(default=2, ge=1)
    #: "auto" picks CUDA when available, else CPU.
    asr_device: str = "auto"

    # ── Limits ────────────────────────────────────────────────────────────────
    #: Upload cap, same as OpenRouter's 25 MB.
    max_upload_mb: float = Field(default=25, gt=0)
    #: Longest audio accepted, in seconds. NeMo decodes the whole clip in one
    #: pass, so very long audio can run out of GPU memory. 0 disables the check.
    max_audio_seconds: float = Field(default=600, ge=0)
    #: Requests allowed to wait for the model while one is running. Past this the
    #: server answers 503 with Retry-After instead of queueing forever.
    max_queue: int = Field(default=16, ge=0)

    # ── HTTP ──────────────────────────────────────────────────────────────────
    #: Bearer keys accepted on /api/v1/*. Empty disables auth (local dev only).
    api_keys: CommaList = []
    cors_origins: CommaList = ["*"]
    log_level: str = "info"

    @field_validator("api_keys", "cors_origins", mode="before")
    @classmethod
    def _split_commas(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
