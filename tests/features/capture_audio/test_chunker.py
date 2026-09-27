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
        emitted.extend(chunker.push(SPEECH))
    for _ in range(30):
        emitted.extend(chunker.push(SILENCE))
    assert len(emitted) == 1
    duration = len(emitted[0]) / 2 / 16000
    assert duration > 0.25


def test_short_noise_is_dropped() -> None:
    chunker = _chunker()
    emitted: list[bytes] = []
    for _ in range(4):
        emitted.extend(chunker.push(SPEECH))
    for _ in range(30):
        emitted.extend(chunker.push(SILENCE))
    assert emitted == []


def test_set_min_silence_ms_changes_cut_point() -> None:
    chunker = SpeechChunker(
        sample_rate=16000,
        frame_samples=FRAME,
        preroll_ms=0,
        min_silence_ms=800,
        max_utterance_s=12,
        min_speech_ms=250,
        is_speech=_energy_speech,
    )
    for _ in range(16):
        assert chunker.push(SPEECH) == []
    for _ in range(10):
        assert chunker.push(SILENCE) == []
    chunker.set_min_silence_ms(250)
    emitted: list[bytes] = []
    for _ in range(10):
        emitted = chunker.push(SILENCE)
        if emitted:
            break
    assert len(emitted) == 1


def test_two_pauses_emit_on_the_second_pause() -> None:
    chunker = _chunker()
    chunker.set_pauses_to_cut(2)
    emitted: list[bytes] = []

    def push_many(frame: bytes, count: int) -> None:
        for _ in range(count):
            emitted.extend(chunker.push(frame))

    push_many(SPEECH, 16)
    push_many(SILENCE, 30)
    assert emitted == []
    push_many(SPEECH, 16)
    push_many(SILENCE, 30)
    assert len(emitted) == 2
    assert all(len(pcm) / 2 / 16000 > 0.25 for pcm in emitted)


def test_flush_emits_a_phrase_held_for_a_second_pause() -> None:
    chunker = _chunker()
    chunker.set_pauses_to_cut(2)
    for _ in range(16):
        assert chunker.push(SPEECH) == []
    for _ in range(30):
        assert chunker.push(SILENCE) == []
    held = chunker.flush()
    assert len(held) == 1
    assert len(held[0]) / 2 / 16000 > 0.25


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
    emitted: list[bytes] = []
    for _ in range(80):
        emitted = chunker.push(SPEECH)
        if emitted:
            break
    assert len(emitted) == 1
    assert len(emitted[0]) / 2 / 16000 >= 0.9
