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
        self.unloaded = False

    def transcribe(self, wav_path: str, *, timestamps: bool) -> Transcript:
        self.calls.append((wav_path, timestamps))
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
    def make(**overrides) -> TestClient:
        settings = Settings(_env_file=None, asr_preload="none", **overrides)

        def loader(spec):
            engine = FakeEngine(spec.id)
            engines[spec.id] = engine
            return engine

        return TestClient(create_app(settings, loader=loader))

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
