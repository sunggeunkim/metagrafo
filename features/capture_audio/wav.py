from __future__ import annotations

import wave
from collections.abc import Iterator
from pathlib import Path

from features.capture_audio.pcm import downmix_to_mono, resample_s16le

TARGET_RATE = 16000
FRAME_SAMPLES = 512


def frames_from_wav(
    path: Path | str,
    *,
    min_silence_ms: int,
    sample_rate: int = TARGET_RATE,
    frame_samples: int = FRAME_SAMPLES,
) -> Iterator[bytes]:
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        src_rate = wf.getframerate()
        width = wf.getsampwidth()
        if width != 2:
            raise ValueError(f"wav must be 16-bit PCM, got {width * 8}-bit")
        raw = wf.readframes(wf.getnframes())
    mono = downmix_to_mono(raw, channels)
    pcm = resample_s16le(mono, src_rate=src_rate, dst_rate=sample_rate)
    step = frame_samples * 2
    leftover = len(pcm) % step
    if leftover:
        pcm = pcm + b"\x00" * (step - leftover)
    for offset in range(0, len(pcm), step):
        yield pcm[offset : offset + step]
    if min_silence_ms <= 0:
        return
    frame_ms = frame_samples / sample_rate * 1000
    silence_frames = max(1, int(round(min_silence_ms / frame_ms)))
    silent = b"\x00" * step
    for _ in range(silence_frames):
        yield silent
