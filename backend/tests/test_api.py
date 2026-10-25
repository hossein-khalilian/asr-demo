import base64

from tests.conftest import make_wav

DEFAULT = "hsekhalilian/speech-to-text-finetuning-01"


def test_models_list_has_openrouter_shape(client):
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    body = res.json()
    assert body["total_count"] == len(body["data"]) > 0
    assert body["links"] == {"next": None}
    model = body["data"][0]
    assert model["id"] == DEFAULT
    assert model["architecture"]["output_modalities"] == ["transcription"]
    assert model["pricing"]["prompt"] == "0"
    assert model["hugging_face_id"] == "hsekhalilian/Speech_To_Text_Finetuning_01"


def test_models_modality_filter(client):
    assert client.get("/api/v1/models?output_modalities=text").json()["data"] == []
    assert client.get("/api/v1/models?output_modalities=transcription").json()["data"]
    assert client.get("/api/v1/models?output_modalities=all").json()["data"]


def test_get_model_by_slug(client):
    assert client.get(f"/api/v1/models/{DEFAULT}").json()["data"]["id"] == DEFAULT
    res = client.get("/api/v1/models/nope/missing")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == 404


def test_multipart_transcription(client, wav_bytes, engines):
    res = client.post(
        "/api/v1/audio/transcriptions",
        data={"model": DEFAULT},
        files={"file": ("sample.wav", wav_bytes, "audio/wav")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["text"] == "سلام دنیا"
    assert body["usage"] == {"seconds": 1.0, "cost": 0.0}
    assert "segments" not in body
    assert res.headers["x-generation-id"].startswith("gen-")
    assert engines[DEFAULT].calls[0][1] is False


def test_json_base64_transcription(client, wav_bytes):
    res = client.post(
        "/api/v1/audio/transcriptions",
        json={
            "model": DEFAULT,
            "input_audio": {"data": base64.b64encode(wav_bytes).decode(), "format": "wav"},
            # OpenRouter routing fields are ignored, not rejected.
            "provider": {"zdr": True},
            "session_id": "abc",
        },
    )
    assert res.status_code == 200, res.text
    assert res.json()["text"] == "سلام دنیا"


def test_raw_pcm_format(client):
    pcm = make_wav(0.5)[44:]  # strip the RIFF header
    res = client.post(
        "/api/v1/audio/transcriptions",
        json={"input_audio": {"data": base64.b64encode(pcm).decode(), "format": "pcm"}},
    )
    assert res.status_code == 200, res.text
    assert res.json()["usage"]["seconds"] == 0.5


def test_model_name_aliases(client, wav_bytes):
    for name in ("hsekhalilian/Speech_To_Text_Finetuning_01", None):
        data = {"model": name} if name else {}
        res = client.post(
            "/api/v1/audio/transcriptions",
            data=data,
            files={"file": ("a.wav", wav_bytes)},
        )
        assert res.status_code == 200, res.text
        assert res.headers["x-model"] == DEFAULT


def test_verbose_json_with_words(client, wav_bytes):
    res = client.post(
        "/api/v1/audio/transcriptions",
        data={"response_format": "verbose_json", "timestamp_granularities[]": ["segment", "word"]},
        files={"file": ("a.wav", wav_bytes)},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["duration"] == 1.0
    assert body["language"] == "fa"
    assert body["segments"] == [{"id": 0, "start": 0.0, "end": 0.5, "text": "سلام دنیا"}]
    assert [w["word"] for w in body["words"]] == ["سلام", "دنیا"]


def test_unknown_model(client, wav_bytes):
    res = client.post(
        "/api/v1/audio/transcriptions",
        data={"model": "openai/whisper-1"},
        files={"file": ("a.wav", wav_bytes)},
    )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == 400
    assert DEFAULT in res.json()["error"]["metadata"]["available"]


def test_rejects_unsupported_options(client, wav_bytes):
    def post(**data):
        return client.post(
            "/api/v1/audio/transcriptions", data=data, files={"file": ("a.wav", wav_bytes)}
        )

    assert post(response_format="srt").status_code == 400
    assert post(language="en").status_code == 400
    assert post(language="fa-IR").status_code == 200
    assert post(diarize="true", response_format="verbose_json").status_code == 400


def test_undecodable_audio(client):
    res = client.post(
        "/api/v1/audio/transcriptions",
        files={"file": ("a.mp3", b"definitely not audio")},
    )
    assert res.status_code == 400
    assert "decode" in res.json()["error"]["message"]


def test_bad_base64_and_url_input(client):
    bad = client.post(
        "/api/v1/audio/transcriptions",
        json={"input_audio": {"data": "!!!", "format": "wav"}},
    )
    assert bad.status_code == 400
    url = client.post(
        "/api/v1/audio/transcriptions",
        json={"input_audio": {"url": "https://example.com/a.wav"}},
    )
    assert url.status_code == 400


def test_upload_limit(make_client):
    with make_client(max_upload_mb=0.01) as client:
        res = client.post(
            "/api/v1/audio/transcriptions",
            files={"file": ("a.wav", make_wav(2.0))},
        )
    assert res.status_code == 413
    assert res.json()["error"]["code"] == 413


def test_audio_length_limit(make_client):
    with make_client(max_audio_seconds=0.5) as client:
        res = client.post(
            "/api/v1/audio/transcriptions",
            files={"file": ("a.wav", make_wav(1.0))},
        )
    assert res.status_code == 400
    assert "limit" in res.json()["error"]["message"]


def test_api_keys(make_client, wav_bytes):
    with make_client(api_keys=["sk-test"]) as client:
        assert client.get("/api/v1/models").status_code == 401
        bad = client.get("/api/v1/models", headers={"Authorization": "Bearer nope"})
        assert bad.status_code == 401
        assert bad.json()["error"]["message"] == "Invalid API key."
        ok = client.get("/api/v1/models", headers={"Authorization": "Bearer sk-test"})
        assert ok.status_code == 200
        # Health stays open for container healthchecks.
        assert client.get("/health").status_code == 200


def test_lru_eviction(make_client, wav_bytes, engines):
    other = "nvidia/stt-fa-fastconformer-hybrid-large"
    third = "hsekhalilian/speech-to-text-rnnt-finetuned"
    with make_client(asr_max_loaded_models=2) as client:
        for model in (DEFAULT, other, third):
            res = client.post(
                "/api/v1/audio/transcriptions",
                data={"model": model},
                files={"file": ("a.wav", wav_bytes)},
            )
            assert res.status_code == 200
        loaded = client.get("/health").json()["loaded_models"]
    assert loaded == [other, third]
    assert engines[DEFAULT].unloaded


def test_openai_sdk_compat(client, wav_bytes):
    """The exact request abrisham-stt sends for a dedicated ASR model."""
    openai = __import__("pytest").importorskip("openai")
    sdk = openai.OpenAI(base_url="http://testserver/api/v1", api_key="x", http_client=client)
    result = sdk.audio.transcriptions.create(
        model=DEFAULT, file=("chunk.wav", wav_bytes), language="fa"
    )
    assert result.text == "سلام دنیا"


def test_segments_from_words():
    from app.asr.engine import segments_from_words

    words = [
        (0.0, 0.4, "a"),
        (1.2, 1.6, "b"),  # 0.8s pause, but the segment is still short: no cut
        (1.7, 3.5, "c"),
        (4.2, 4.5, "d"),  # 0.7s pause after a 3.5s segment: cut
        (6.5, 6.8, "e"),  # 2s pause: always cut
        (6.9, 18.0, "f"),
        (18.1, 18.8, "g"),  # would run the segment past 12s: cut
    ]
    assert segments_from_words(words) == [
        (0.0, 3.5, "a b c"),
        (4.2, 4.5, "d"),
        (6.5, 18.0, "e f"),
        (18.1, 18.8, "g"),
    ]
    assert segments_from_words([]) == []


PRIVATE = "hsekhalilian/Speech_To_Text_Finetuning_01"
NVIDIA = "nvidia/stt-fa-fastconformer-hybrid-large"


class RepositoryNotFoundError(Exception):
    """Stands in for huggingface_hub's; matched by class name."""


class GatedRepoError(RepositoryNotFoundError):
    pass


def _wait_for_probe(client):
    import time

    for _ in range(100):
        if client.get("/health").json()["unavailable_models"]:
            return
        time.sleep(0.01)
    raise AssertionError("startup probe never ran")


def test_unavailable_models_are_reported(make_client, wav_bytes):
    def checker(spec):
        return "private or missing on Hugging Face" if spec.source == PRIVATE else None

    with make_client(access_checker=checker) as client:
        _wait_for_probe(client)
        models = {m["id"]: m for m in client.get("/api/v1/models").json()["data"]}
        assert models[DEFAULT]["available"] is False
        assert "private" in models[DEFAULT]["unavailable_reason"]
        assert models[NVIDIA]["available"] is True
        assert models[NVIDIA]["unavailable_reason"] is None

        res = client.post(
            "/api/v1/audio/transcriptions",
            data={"model": DEFAULT},
            files={"file": ("a.wav", wav_bytes)},
        )
        assert res.status_code == 503
        assert "unavailable on this server" in res.json()["error"]["message"]
        assert "retry-after" not in res.headers


def test_unavailable_model_does_not_evict_a_working_one(make_client, wav_bytes, engines):
    def checker(spec):
        return "private" if spec.source == PRIVATE else None

    with make_client(access_checker=checker, asr_max_loaded_models=1) as client:
        _wait_for_probe(client)
        post = lambda model: client.post(  # noqa: E731
            "/api/v1/audio/transcriptions",
            data={"model": model},
            files={"file": ("a.wav", wav_bytes)},
        )
        assert post(NVIDIA).status_code == 200
        assert post(DEFAULT).status_code == 503
        assert client.get("/health").json()["loaded_models"] == [NVIDIA]
    assert not engines[NVIDIA].unloaded


def test_auth_failure_on_load_marks_model_unavailable(make_client, wav_bytes):
    error = RepositoryNotFoundError("401 Client Error ... Invalid username or password.")
    with make_client(load_error={PRIVATE: error}) as client:
        res = client.post(
            "/api/v1/audio/transcriptions",
            data={"model": DEFAULT},
            files={"file": ("a.wav", wav_bytes)},
        )
        assert res.status_code == 503
        message = res.json()["error"]["message"]
        assert "HF_TOKEN" in message
        assert "401 Client Error" not in message
        model = client.get(f"/api/v1/models/{DEFAULT}").json()["data"]
        assert model["available"] is False


def test_access_error_classification():
    from app.asr.engine import access_error

    assert "gated" in access_error(GatedRepoError())
    assert "private" in access_error(RepositoryNotFoundError())
    assert access_error(RuntimeError("CUDA out of memory")) is None


def test_timing_headers(client, wav_bytes):
    res = client.post("/api/v1/audio/transcriptions", files={"file": ("a.wav", wav_bytes)})
    assert res.status_code == 200
    timings = {name: int(res.headers[f"x-{name}-ms"]) for name in ("queue", "load", "inference")}
    assert all(value >= 0 for value in timings.values())
    assert int(res.headers["x-processing-ms"]) >= timings["inference"]


def test_error_responses_are_logged_with_reason(client, caplog):
    caplog.set_level("INFO", logger="app.errors")
    client.post("/api/v1/audio/transcriptions", files={"file": ("a.mp3", b"not audio")})
    client.get("/api/v1/models/nope/missing")
    lines = [(r.levelname, r.getMessage()) for r in caplog.records if r.name == "app.errors"]
    assert lines[0][0] == "WARNING"
    assert lines[0][1].startswith("400 POST /api/v1/audio/transcriptions: Could not decode")
    # A client probing for a model id is routine, not a warning.
    assert lines[1] == (
        "INFO",
        "404 GET /api/v1/models/nope/missing: Model 'nope/missing' not found.",
    )


def test_unknown_framework_log_level_is_ignored(caplog):
    from app.asr.engine import quiet_nemo

    quiet_nemo("chatty")  # returns before importing NeMo, which tests do not install
    assert "Unknown ASR_FRAMEWORK_LOG_LEVEL" in caplog.text


def test_nemo_log_filter():
    import logging

    from app.asr.engine import _NemoLevelFilter

    def record(level, path):
        return logging.LogRecord("x", level, path, 1, "msg", None, None)

    nemo_file = (
        "/opt/venv/lib/python3.12/site-packages/nemo/collections/common/data/lhotse/cutset.py"
    )
    keep = _NemoLevelFilter(logging.ERROR)
    assert not keep.filter(record(logging.WARNING, nemo_file))
    assert not keep.filter(record(logging.INFO, nemo_file))
    assert keep.filter(record(logging.ERROR, nemo_file))
    # Our own logs and other libraries are untouched.
    assert keep.filter(record(logging.INFO, "/app/app/routes/transcriptions.py"))


WHISPER = "c1tech/whisper-base-persian"


def test_local_copy_is_never_downloaded(tmp_path):
    from app.asr.engine import check_access
    from app.asr.spec import ModelSpec

    folder = tmp_path / "whisper"
    spec = ModelSpec(
        id="c1tech/x",
        source="C1Tech/whisper_base_persian",
        name="x",
        framework="transformers",
        local_path=str(folder),
    )
    reason = check_access(spec)
    assert reason.startswith("not downloaded yet")
    assert "hf download C1Tech/whisper_base_persian --local-dir" in reason

    folder.mkdir()
    (folder / "README.md").write_text("gated repo: only the card came down")
    assert check_access(spec).startswith("not downloaded yet")
    (folder / "config.json").write_text("{}")
    assert check_access(spec).startswith("download incomplete")
    (folder / "model.safetensors").write_bytes(b"weights")
    assert check_access(spec) is None


def test_undownloaded_whisper_is_listed_unavailable(make_client, tmp_path):
    import json

    from app.asr.engine import check_access

    catalog = tmp_path / "models.json"
    catalog.write_text(
        json.dumps(
            {
                "models": [
                    {"id": DEFAULT, "source": PRIVATE},
                    {
                        "id": WHISPER,
                        "source": "C1Tech/whisper_base_persian",
                        "framework": "transformers",
                        "local_path": str(tmp_path / "not-downloaded"),
                    },
                ]
            }
        )
    )

    # Real check for local copies (disk only); skip the Hub for the rest.
    def checker(spec):
        return check_access(spec) if spec.local_path else None

    with make_client(access_checker=checker, asr_models_file=catalog) as client:
        _wait_for_probe(client)
        model = client.get(f"/api/v1/models/{WHISPER}").json()["data"]
    assert model["available"] is False
    assert "hf download C1Tech/whisper_base_persian" in model["unavailable_reason"]
    assert model["hugging_face_id"] == "C1Tech/whisper_base_persian"


def test_catalog_frameworks():
    from app.asr.spec import load_specs
    from app.settings import BACKEND_DIR

    specs = {s.id: s for s in load_specs(BACKEND_DIR / "config" / "models.json")}
    assert specs[DEFAULT].framework == "nemo"
    whisper = specs[WHISPER]
    assert whisper.framework == "transformers"
    assert dict(whisper.generate) == {"language": "fa", "task": "transcribe"}
    assert whisper.load_from == str(BACKEND_DIR / "models" / "whisper_base_persian")


def test_unknown_framework_is_rejected(tmp_path):
    import json

    import pytest

    from app.asr.spec import load_specs

    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": [{"source": "a/b", "framework": "onnx"}]}))
    with pytest.raises(ValueError, match="framework"):
        load_specs(path)
