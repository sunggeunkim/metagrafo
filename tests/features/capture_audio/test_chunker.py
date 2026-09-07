import numpy as np

from features.capture_audio.chunker import SpeechChunker

FRAME = 512
SPEECH = np.full(FRAME, 8000, dtype=np.int16).tobytes()
SILENCE = np.zeros(FRAME, dtype=np.int16).tobytes()


def _energy_speech(frame: bytes) -> bool:
    return float(np.abs(np.frombuffer(frame, dtype=np.int16)).mean()) > 1000


def _chunker() -> SpeechChunker:
    return SpeechChunker(
        sample_rate=16000,
        frame_samples=FRAME,
        preroll_ms=200,
        min_silence_ms=800,
        max_utterance_s=12,
        min_speech_ms=250,
        is_speech=_energy_speech,
    )


def test_speech_then_silence_emits_one_utterance() -> None:
    chunker = _chunker()
    emitted: list[bytes] = []
    for _ in range(16):
        out = chunker.push(SPEECH)
        if out:
            emitted.append(out)
    for _ in range(30):
        out = chunker.push(SILENCE)
        if out:
            emitted.append(out)
    assert len(emitted) == 1
    duration = len(emitted[0]) / 2 / 16000
    assert duration > 0.25


def test_short_noise_is_dropped() -> None:
    chunker = _chunker()
    emitted: list[bytes] = []
    for _ in range(4):
        out = chunker.push(SPEECH)
        if out:
            emitted.append(out)
    for _ in range(30):
        out = chunker.push(SILENCE)
        if out:
            emitted.append(out)
    assert emitted == []


def test_max_utterance_emits_without_waiting_for_silence() -> None:
    chunker = SpeechChunker(
        sample_rate=16000,
        frame_samples=FRAME,
        preroll_ms=0,
        min_silence_ms=800,
        max_utterance_s=1,
        min_speech_ms=250,
        is_speech=_energy_speech,
    )
    emitted = None
    for _ in range(80):
        emitted = chunker.push(SPEECH)
        if emitted:
            break
    assert emitted is not None
    assert len(emitted) / 2 / 16000 >= 0.9
