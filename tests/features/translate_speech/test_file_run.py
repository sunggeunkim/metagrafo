import wave
from pathlib import Path

import numpy as np

from features.capture_audio.chunker import SpeechChunker
from features.translate_speech.file_run import run_wav

FRAME = 512
SPEECH = np.full(FRAME * 16, 8000, dtype=np.int16)
SILENCE = np.zeros(FRAME * 30, dtype=np.int16)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.astype(np.int16).tobytes())


def _energy_speech(frame: bytes) -> bool:
    return float(np.abs(np.frombuffer(frame, dtype=np.int16)).mean()) > 1000


def _chunker(min_silence_ms: int = 400) -> SpeechChunker:
    return SpeechChunker(
        sample_rate=16000,
        frame_samples=FRAME,
        preroll_ms=200,
        min_silence_ms=min_silence_ms,
        max_utterance_s=12,
        min_speech_ms=250,
        is_speech=_energy_speech,
    )


def test_speech_silence_speech_writes_two_lines(tmp_path: Path) -> None:
    wav_path = tmp_path / "sermon.wav"
    _write_wav(wav_path, np.concatenate([SPEECH, SILENCE, SPEECH]))
    out_path = tmp_path / "sermon.en.txt"
    calls: list[int] = []

    def transcribe(pcm: bytes, *, language: str, task: str, initial_prompt: str) -> str:
        calls.append(len(pcm))
        assert language == "ko"
        assert task == "translate"
        assert initial_prompt == "Jesus"
        return "Hello, everyone."

    n = run_wav(
        wav_path=wav_path,
        out_path=out_path,
        transcribe=transcribe,
        chunker=_chunker(),
        mode="ko_to_en",
        initial_prompt="Jesus",
        min_silence_ms=400,
    )
    assert n == 2
    assert out_path.read_text(encoding="utf-8") == "Hello, everyone.\nHello, everyone.\n"
    assert len(calls) == 2


def test_eof_without_silence_still_emits_last_line(tmp_path: Path) -> None:
    wav_path = tmp_path / "end.wav"
    _write_wav(wav_path, SPEECH)
    out_path = tmp_path / "end.en.txt"

    def transcribe(_pcm: bytes, **_kwargs: object) -> str:
        return "Last line."

    n = run_wav(
        wav_path=wav_path,
        out_path=out_path,
        transcribe=transcribe,
        chunker=_chunker(),
        mode="ko_to_en",
        initial_prompt="",
        min_silence_ms=400,
    )
    assert n == 1
    assert out_path.read_text(encoding="utf-8") == "Last line.\n"


def test_empty_whisper_result_is_skipped(tmp_path: Path) -> None:
    wav_path = tmp_path / "empty.wav"
    _write_wav(wav_path, SPEECH)
    out_path = tmp_path / "empty.en.txt"

    def transcribe(_pcm: bytes, **_kwargs: object) -> str:
        return "  "

    n = run_wav(
        wav_path=wav_path,
        out_path=out_path,
        transcribe=transcribe,
        chunker=_chunker(),
        mode="ko_to_en",
        initial_prompt="",
        min_silence_ms=400,
    )
    assert n == 0
    assert out_path.read_text(encoding="utf-8") == ""
