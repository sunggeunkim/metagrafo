from __future__ import annotations

import numpy as np


def downmix_to_mono(pcm_s16le: bytes, channels: int) -> bytes:
    if channels <= 1:
        return pcm_s16le
    samples = np.frombuffer(pcm_s16le, dtype=np.int16)
    frames = samples.reshape(-1, channels)
    mono = frames.mean(axis=1).astype(np.int16)
    return mono.tobytes()


def resample_s16le(pcm_s16le: bytes, *, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate:
        return pcm_s16le
    samples = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return pcm_s16le
    dst_len = max(1, int(round(samples.size * dst_rate / src_rate)))
    src_x = np.linspace(0.0, 1.0, samples.size, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, dst_len, endpoint=False)
    resampled = np.interp(dst_x, src_x, samples).astype(np.int16)
    return resampled.tobytes()
