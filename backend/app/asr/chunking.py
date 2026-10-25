"""Splitting audio into overlapping chunks and stitching their transcripts.

Chunks overlap so no word is cut in half at a boundary. Each word is then
kept by exactly one chunk: the boundary between chunk i and i+1 is the middle
of their overlap, and a word belongs to the chunk its midpoint falls in. That
needs word timestamps; without them (beam search, Whisper) the transcripts are
joined by dropping the longest run of words a chunk repeats from the end of
the text so far.
"""

from __future__ import annotations

import re
import wave
from dataclasses import dataclass
from pathlib import Path

from app.asr.engine import Transcript

#: A last chunk shorter than this is folded into the one before it: a sliver of
#: audio transcribes badly and its words are already in the overlap.
MIN_LAST_CHUNK_SECONDS = 1.0
#: Longest run of repeated words looked for when joining by text.
MAX_TEXT_OVERLAP_WORDS = 40


@dataclass(frozen=True)
class Chunk:
    path: Path
    offset: float
    duration: float


def split_wav(
    path: Path, chunk_seconds: float, overlap_seconds: float, workdir: Path
) -> list[Chunk]:
    """Cut a mono 16-bit WAV into chunks. One chunk (the file itself) when
    chunking is off or the audio is no longer than a chunk."""
    with wave.open(str(path), "rb") as wav:
        params = wav.getparams()
        frames = wav.readframes(params.nframes)
    rate = params.framerate
    total = params.nframes / rate
    if chunk_seconds <= 0 or total <= chunk_seconds:
        return [Chunk(path, 0.0, total)]

    step = chunk_seconds - overlap_seconds
    starts = [0.0]
    while starts[-1] + chunk_seconds < total:
        starts.append(starts[-1] + step)
    if len(starts) > 1 and total - starts[-1] < MIN_LAST_CHUNK_SECONDS + overlap_seconds:
        starts.pop()

    frame_bytes = params.sampwidth * params.nchannels
    chunks: list[Chunk] = []
    for i, start in enumerate(starts):
        end = total if i == len(starts) - 1 else start + chunk_seconds
        first, last = int(start * rate), int(end * rate)
        chunk_path = workdir / f"chunk-{i:04d}.wav"
        with wave.open(str(chunk_path), "wb") as out:
            out.setparams(params)
            out.writeframes(frames[first * frame_bytes : last * frame_bytes])
        chunks.append(Chunk(chunk_path, round(start, 3), round(end - start, 3)))
    return chunks


def merge(chunks: list[Chunk], transcripts: list[Transcript], overlap_seconds: float) -> Transcript:
    """One transcript for the whole audio, with times relative to its start."""
    if len(chunks) == 1:
        return transcripts[0]
    bounds = _owned_ranges(chunks, overlap_seconds)
    if all(t.words or not t.text for t in transcripts):
        return _merge_by_time(chunks, transcripts, bounds)
    return _merge_by_text(chunks, transcripts, bounds)


def _owned_ranges(chunks: list[Chunk], overlap: float) -> list[tuple[float, float]]:
    """The span of the audio each chunk is responsible for."""
    ranges = []
    for i, chunk in enumerate(chunks):
        lo = chunk.offset + overlap / 2 if i > 0 else 0.0
        hi = chunk.offset + chunk.duration - (overlap / 2 if i < len(chunks) - 1 else 0.0)
        ranges.append((lo, hi))
    return ranges


def _merge_by_time(
    chunks: list[Chunk], transcripts: list[Transcript], bounds: list[tuple[float, float]]
) -> Transcript:
    words: list[tuple[float, float, str]] = []
    last = len(chunks) - 1
    for i, (chunk, transcript, (lo, hi)) in enumerate(
        zip(chunks, transcripts, bounds, strict=True)
    ):
        for start, end, word in transcript.words:
            start, end = start + chunk.offset, end + chunk.offset
            middle = (start + end) / 2
            if lo <= middle < hi or (i == last and middle >= lo):
                words.append((round(start, 3), round(end, 3), word))
    # Segments are rebuilt from the words by the caller (pause-based).
    return Transcript(text=" ".join(w for _, _, w in words), words=words)


def _merge_by_text(
    chunks: list[Chunk], transcripts: list[Transcript], bounds: list[tuple[float, float]]
) -> Transcript:
    merged: list[str] = []
    segments: list[tuple[float, float, str]] = []
    for chunk, transcript, (lo, hi) in zip(chunks, transcripts, bounds, strict=True):
        incoming = transcript.text.split()
        repeated = _repeated_prefix(merged, incoming)
        piece = incoming[repeated:]
        merged.extend(piece)
        if transcript.segments:
            # Whisper: keep the chunk's own segments that start in its range.
            for start, end, text in transcript.segments:
                start, end = start + chunk.offset, end + chunk.offset
                if lo <= start < hi:
                    segments.append((round(start, 3), round(min(end, hi), 3), text))
        elif piece:
            segments.append((round(lo, 3), round(hi, 3), " ".join(piece)))
    return Transcript(text=" ".join(merged), segments=segments)


_NOT_WORD = re.compile(r"[‌‏\W_]+")


def _norm(word: str) -> str:
    # Ignore punctuation and Persian zero-width non-joiners when comparing.
    return _NOT_WORD.sub("", word).casefold()


def _repeated_prefix(merged: list[str], incoming: list[str]) -> int:
    """How many words at the start of ``incoming`` repeat the end of ``merged``."""
    longest = min(len(merged), len(incoming), MAX_TEXT_OVERLAP_WORDS)
    tail = [_norm(w) for w in merged[-longest:]] if longest else []
    head = [_norm(w) for w in incoming[:longest]]
    for k in range(longest, 0, -1):
        if tail[-k:] == head[:k]:
            return k
    return 0
