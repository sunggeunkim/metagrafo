from __future__ import annotations

import wave
from collections.abc import Iterator
from pathlib import Path

from features.capture_audio.pcm import downmix_to_mono, resample_s16le

TARGET_RATE = 16000
FRAME_SAMPLES = 512


def _source_span(n_frames: int, rate: int, start_s: float, end_s: float | None) -> tuple[int, int]:
    start = 0 if start_s <= 0 else min(n_frames, int(round(start_s * rate)))
    if end_s is None:
        end = n_frames
    elif end_s <= 0:
        end = 0
    else:
        end = min(n_frames, int(round(end_s * rate)))
    if end < start:
        end = start
    return start, end


def frames_from_wav(
    path: Path | str,
    *,
    min_silence_ms: int,
    sample_rate: int = TARGET_RATE,
    frame_samples: int = FRAME_SAMPLES,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> Iterator[bytes]:
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        src_rate = wf.getframerate()
        width = wf.getsampwidth()
        if width != 2:
            raise ValueError(f"wav must be 16-bit PCM, got {width * 8}-bit")
        start, end = _source_span(wf.getnframes(), src_rate, start_s, end_s)
        if start:
            wf.setpos(start)
        raw = wf.readframes(end - start)
    mono = downmix_to_mono(raw, channels)
    pcm = resample_s16le(mono, src_rate=src_rate, dst_rate=sample_rate)
    step = frame_samples * 2
    leftover = len(pcm) % step
    if leftover:
        pcm = pcm + b"\x00" * (step - leftover)
    for offset in range(0, len(pcm), step):
        yield pcm[offset : offset + step]
    # SpeechChunker emits only after trailing silence; pad at least one EOF frame
    # even when min_silence_ms is nonpositive (chunker uses max(1, …)).
    frame_ms = frame_samples / sample_rate * 1000
    silence_frames = max(1, int(round(max(0, min_silence_ms) / frame_ms)))
    silent = b"\x00" * step
    for _ in range(silence_frames):
        yield silent
