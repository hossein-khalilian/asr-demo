"""Tuning options: chunking, stitching, decoding settings and audio filters."""

import json
import wave

import pytest

from app.asr.chunking import Chunk, merge, split_wav
from app.asr.engine import DecodeOptions, Transcript
from tests.conftest import make_wav

DEFAULT = "hsekhalilian/speech-to-text-finetuning-01"


def _wav(tmp_path, seconds):
    path = tmp_path / "audio.wav"
    path.write_bytes(make_wav(seconds))
    return path


def _seconds(path):
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


# ── split_wav ───────────────────────────────────────────────────────────────


def test_no_chunking_returns_the_file(tmp_path):
    path = _wav(tmp_path, 12)
    assert split_wav(path, 0, 0, tmp_path) == [Chunk(path, 0.0, 12.0)]
    assert split_wav(path, 30, 2, tmp_path) == [Chunk(path, 0.0, 12.0)]  # shorter than a chunk


def test_chunks_overlap_and_cover_the_audio(tmp_path):
    chunks = split_wav(_wav(tmp_path, 25), 10, 2, tmp_path)
    assert [(c.offset, c.duration) for c in chunks] == [(0.0, 10.0), (8.0, 10.0), (16.0, 9.0)]
    assert [round(_seconds(c.path), 3) for c in chunks] == [10.0, 10.0, 9.0]


def test_a_sliver_at_the_end_joins_the_previous_chunk(tmp_path):
    # 10s chunks, 2s overlap over 18.5s: a third chunk would start at 16s and
    # hold only 0.5s beyond the overlap, so the second runs to the end.
    chunks = split_wav(_wav(tmp_path, 18.5), 10, 2, tmp_path)
    assert [(c.offset, c.duration) for c in chunks] == [(0.0, 10.0), (8.0, 10.5)]


# ── merge ───────────────────────────────────────────────────────────────────


def test_merge_by_word_times_keeps_each_word_once():
    chunks = [Chunk(None, 0.0, 10.0), Chunk(None, 8.0, 10.0)]
    # The word at 8.4–8.8s (absolute) is heard by both chunks.
    first = Transcript("a b c", words=[(1.0, 1.4, "a"), (7.0, 7.5, "b"), (8.4, 8.8, "c")])
    second = Transcript("c d", words=[(0.4, 0.8, "c"), (3.0, 3.5, "d")])
    merged = merge(chunks, [first, second], overlap_seconds=2)
    assert merged.text == "a b c d"
    assert [w for *_, w in merged.words] == ["a", "b", "c", "d"]
    assert merged.words[-1] == (11.0, 11.5, "d")  # shifted to absolute time


def test_merge_by_text_drops_the_repeated_words():
    chunks = [Chunk(None, 0.0, 10.0), Chunk(None, 8.0, 10.0)]
    # No word times (beam search): join on the longest repeated run, ignoring
    # punctuation and zero-width non-joiners.
    first = Transcript("این کتاب راهنمای واجبه، برای هر")
    second = Transcript("برای هر مهندسی‌ها چرا")
    merged = merge(chunks, [first, second], overlap_seconds=2)
    assert merged.text == "این کتاب راهنمای واجبه، برای هر مهندسی‌ها چرا"
    assert merged.segments == [
        (0.0, 9.0, "این کتاب راهنمای واجبه، برای هر"),
        (9.0, 18.0, "مهندسی‌ها چرا"),
    ]


# ── API ─────────────────────────────────────────────────────────────────────


def _post(client, wav, *, provider=None, **data):
    if provider is not None:
        data["provider"] = json.dumps({"options": {"asr-demo": provider}})
    return client.post("/api/v1/audio/transcriptions", data=data, files={"file": ("a.wav", wav)})


def test_untuned_request_is_one_pass_with_default_decoding(client, engines):
    res = _post(client, make_wav(12))
    assert res.status_code == 200
    assert res.headers["x-chunks"] == "1"
    assert engines[DEFAULT].decodes == [DecodeOptions()]


def test_chunked_request_over_multipart(client, engines):
    res = _post(
        client,
        make_wav(25),
        provider={"chunk_seconds": 10, "overlap_seconds": 2, "decoder": "ctc", "beam_size": 4},
    )
    assert res.status_code == 200, res.text
    assert res.headers["x-chunks"] == "3"
    engine = engines[DEFAULT]
    assert len(engine.calls) == 3
    assert all(ts for _, ts in engine.calls)  # word times for stitching
    assert engine.decodes[0] == DecodeOptions(decoder="ctc", beam_size=4)


def test_tuning_over_json_body(client, engines):
    import base64

    res = client.post(
        "/api/v1/audio/transcriptions",
        json={
            "input_audio": {"data": base64.b64encode(make_wav(25)).decode(), "format": "wav"},
            "provider": {"options": {"asr-demo": {"chunk_seconds": 10}, "other": {"x": 1}}},
        },
    )
    assert res.status_code == 200, res.text
    assert res.headers["x-chunks"] == "3"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"chunk_seconds": 3}, "at least 5"),
        ({"chunk_seconds": 10, "overlap_seconds": 5}, "less than half"),
        ({"beam_size": 0}, "beam_size"),
        ({"decoder": "attention"}, "decoder"),
        ({"chunk_size": 10}, "chunk_size"),  # typo: unknown keys are errors
    ],
)
def test_invalid_tuning_options(client, options, message):
    res = _post(client, make_wav(1), provider=options)
    assert res.status_code == 400
    error = res.json()["error"]["message"]
    assert error.startswith("provider.options.asr-demo")
    assert message in error


def test_provider_must_be_json_in_multipart(client):
    res = client.post(
        "/api/v1/audio/transcriptions",
        data={"provider": "chunk_seconds=10"},
        files={"file": ("a.wav", make_wav(1))},
    )
    assert res.status_code == 400
    assert "JSON-encoded" in res.json()["error"]["message"]


def test_decoder_is_rejected_for_whisper(client):
    res = _post(
        client, make_wav(1), model="c1tech/whisper-base-persian", provider={"decoder": "ctc"}
    )
    assert res.status_code == 400
    assert "decoder" in res.json()["error"]["message"]


def test_temperature_reaches_whisper_only(client, engines):
    assert _post(client, make_wav(1), temperature="0.4").status_code == 200
    assert engines[DEFAULT].decodes[-1].temperature is None  # NeMo ignores it
    whisper = "c1tech/whisper-base-persian"
    assert _post(client, make_wav(1), model=whisper, temperature="0.4").status_code == 200
    assert engines[whisper].decodes[-1].temperature == 0.4


def test_audio_filters_run_through_ffmpeg(client):
    res = _post(
        client, make_wav(3), provider={"normalize": True, "denoise": True, "highpass": True}
    )
    assert res.status_code == 200, res.text
    assert res.json()["usage"]["seconds"] == 3.0  # filters keep the length


def test_models_advertise_their_parameters(client):
    models = {m["id"]: m for m in client.get("/api/v1/models").json()["data"]}
    nemo = models[DEFAULT]["supported_parameters"]
    whisper = models["c1tech/whisper-base-persian"]["supported_parameters"]
    assert {"chunk_seconds", "overlap_seconds", "beam_size", "decoder"} <= set(nemo)
    assert "temperature" not in nemo
    assert "temperature" in whisper and "decoder" not in whisper


WHISPER = "c1tech/whisper-base-persian"


def _chunk_lengths(engine):
    return engine.durations


def test_whisper_is_always_chunked_to_its_window(client, engines):
    # 70s with no chunking asked for: 30s windows overlapping by the default 2s.
    res = _post(client, make_wav(70), model=WHISPER)
    assert res.status_code == 200, res.text
    assert res.headers["x-chunks"] == "3"
    assert _chunk_lengths(engines[WHISPER]) == [30.0, 30.0, 14.0]


def test_whisper_caps_a_longer_requested_chunk(client, engines):
    res = _post(
        client, make_wav(70), model=WHISPER, provider={"chunk_seconds": 60, "overlap_seconds": 4}
    )
    assert res.status_code == 200, res.text
    assert max(_chunk_lengths(engines[WHISPER])) == 30.0


def test_short_whisper_audio_is_one_pass(client, engines):
    assert _post(client, make_wav(20), model=WHISPER).headers["x-chunks"] == "1"


def test_nemo_has_no_window(client, engines):
    assert _post(client, make_wav(70)).headers["x-chunks"] == "1"
    models = {m["id"]: m for m in client.get("/api/v1/models").json()["data"]}
    assert models[DEFAULT].get("max_chunk_seconds") is None
    assert models[WHISPER]["max_chunk_seconds"] == 30
