import io
import math
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from app.asr.engine import Transcript
from app.main import create_app
from app.settings import Settings


class FakeEngine:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.calls: list[tuple[str, bool]] = []
        self.decodes: list = []
        self.durations: list[float] = []
        self.unloaded = False

    def transcribe(self, wav_path: str, *, timestamps: bool, decode=None) -> Transcript:
        self.calls.append((wav_path, timestamps))
        self.decodes.append(decode)
        with wave.open(wav_path, "rb") as wav:  # the request deletes its files afterwards
            self.durations.append(round(wav.getnframes() / wav.getframerate(), 1))
        if not timestamps:
            return Transcript(text="سلام دنیا")
        return Transcript(
            text="سلام دنیا",
            segments=[(0.0, 0.5, "سلام دنیا")],
            words=[(0.0, 0.2, "سلام"), (0.25, 0.5, "دنیا")],
        )

    def unload(self) -> None:
        self.unloaded = True


@pytest.fixture
def engines() -> dict[str, FakeEngine]:
    return {}


@pytest.fixture
def make_client(engines):
    def make(*, access_checker=lambda source: None, load_error=None, **overrides) -> TestClient:
        """``load_error`` maps a model's source to the exception loading it raises."""
        settings = Settings(_env_file=None, asr_preload="none", **overrides)

        def loader(spec):
            if load_error and spec.source in load_error:
                raise load_error[spec.source]
            engine = FakeEngine(spec.id)
            engines[spec.id] = engine
            return engine

        return TestClient(create_app(settings, loader=loader, access_checker=access_checker))

    return make


@pytest.fixture
def client(make_client) -> TestClient:
    with make_client() as c:
        yield c


def make_wav(seconds: float = 1.0, rate: int = 16_000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
            for i in range(int(seconds * rate))
        )
        wav.writeframes(frames)
    return buf.getvalue()


@pytest.fixture
def wav_bytes() -> bytes:
    return make_wav()
