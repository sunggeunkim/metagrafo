import wave
from pathlib import Path

import numpy as np

from features.capture_audio.wav import frames_from_wav

FRAME = 512


def _write_wav(
    path: Path,
    samples: np.ndarray,
    *,
    rate: int = 16000,
    channels: int = 1,
) -> None:
    pcm = samples.astype(np.int16)
    if channels == 2:
        if pcm.ndim == 1:
            raise AssertionError("stereo needs shape (n, 2)")
        pcm = pcm.reshape(-1, 2)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def test_frames_are_512_samples_at_16k(tmp_path: Path) -> None:
    samples = np.full(FRAME * 4, 1000, dtype=np.int16)
    path = tmp_path / "mono.wav"
    _write_wav(path, samples)
    frames = list(frames_from_wav(path, min_silence_ms=400))
    assert all(len(frame) == FRAME * 2 for frame in frames)
    speech_frames = frames[:4]
    assert all(
        int(np.frombuffer(frame, dtype=np.int16).mean()) == 1000 for frame in speech_frames
    )


def test_stereo_48k_downmixes_and_resamples(tmp_path: Path) -> None:
    n = 48000
    stereo = np.column_stack(
        [np.full(n, 1000, dtype=np.int16), np.full(n, 3000, dtype=np.int16)]
    )
    path = tmp_path / "stereo48k.wav"
    _write_wav(path, stereo, rate=48000, channels=2)
    frames = list(frames_from_wav(path, min_silence_ms=0))
    assert all(len(frame) == FRAME * 2 for frame in frames)
    first = np.frombuffer(frames[0], dtype=np.int16)
    assert int(first[0]) == 2000
    speech_samples = sum(len(np.frombuffer(f, dtype=np.int16)) for f in frames)
    assert speech_samples >= 16000


def test_eof_appends_trailing_silence_frames(tmp_path: Path) -> None:
    samples = np.full(FRAME * 2, 8000, dtype=np.int16)
    path = tmp_path / "speech.wav"
    _write_wav(path, samples)
    frames = list(frames_from_wav(path, min_silence_ms=320))
    silent = [f for f in frames if int(np.abs(np.frombuffer(f, dtype=np.int16)).mean()) == 0]
    assert len(silent) == 10
