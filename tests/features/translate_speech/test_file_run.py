import wave
from pathlib import Path

import numpy as np
import pytest

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


def test_gemini_engine_writes_the_translation_not_the_transcript(tmp_path: Path) -> None:
    wav_path = tmp_path / "sermon.wav"
    _write_wav(wav_path, SPEECH)
    out_path = tmp_path / "sermon.en.txt"

    def recognize(
        _pcm: bytes,
        *,
        language: str,
        task: str,
        initial_prompt: str,
        condition_on_previous_text: bool = False,
    ) -> str:
        if language == "ko" and task == "transcribe" and initial_prompt == "":
            return "기도"
        return "God is good."

    def translate(source: str, *, prior_english: list[str]) -> str:
        if source == "기도" and prior_english == []:
            return "Prayer"
        return "bad context"

    cues = run_wav(
        wav_path=wav_path,
        out_path=out_path,
        transcribe=recognize,
        chunker=_chunker(),
        mode="ko_to_en",
        min_silence_ms=400,
        engine="gemini",
        translate=translate,
    )
    assert len(cues) == 1
    assert out_path.read_text(encoding="utf-8") == "Prayer\n"


def test_two_pauses_write_one_line_per_phrase(tmp_path: Path) -> None:
    wav_path = tmp_path / "sermon.wav"
    _write_wav(wav_path, np.concatenate([SPEECH, SILENCE, SPEECH]))
    out_path = tmp_path / "sermon.en.txt"
    calls: list[int] = []

    def transcribe(pcm: bytes, **_kwargs: object) -> str:
        calls.append(len(pcm))
        return f"Phrase {len(calls)}."

    chunker = _chunker()
    chunker.set_pauses_to_cut(2)
    cues = run_wav(
        wav_path=wav_path,
        out_path=out_path,
        transcribe=transcribe,
        chunker=chunker,
        mode="ko_to_en",
        min_silence_ms=400,
    )
    assert len(cues) == 2
    assert len(calls) == 2
    assert out_path.read_text(encoding="utf-8") == "Phrase 1.\nPhrase 2.\n"


def test_speech_silence_speech_writes_two_lines(tmp_path: Path) -> None:
    wav_path = tmp_path / "sermon.wav"
    _write_wav(wav_path, np.concatenate([SPEECH, SILENCE, SPEECH]))
    out_path = tmp_path / "sermon.en.txt"
    calls: list[int] = []

    def transcribe(
        pcm: bytes,
        *,
        language: str,
        task: str,
        initial_prompt: str,
        condition_on_previous_text: bool = False,
    ) -> str:
        calls.append(len(pcm))
        assert language == "ko"
        assert task == "translate"
        assert initial_prompt == ""
        return "Hello, everyone."

    cues = run_wav(
        wav_path=wav_path,
        out_path=out_path,
        transcribe=transcribe,
        chunker=_chunker(),
        mode="ko_to_en",
        min_silence_ms=400,
    )
    assert len(cues) == 2
    assert out_path.read_text(encoding="utf-8") == "Hello, everyone.\nHello, everyone.\n"
    assert len(calls) == 2


def test_eof_without_silence_still_emits_last_line(tmp_path: Path) -> None:
    wav_path = tmp_path / "end.wav"
    _write_wav(wav_path, SPEECH)
    out_path = tmp_path / "end.en.txt"

    def transcribe(_pcm: bytes, **_kwargs: object) -> str:
        return "Last line."

    cues = run_wav(
        wav_path=wav_path,
        out_path=out_path,
        transcribe=transcribe,
        chunker=_chunker(),
        mode="ko_to_en",
        min_silence_ms=400,
    )
    assert len(cues) == 1
    assert out_path.read_text(encoding="utf-8") == "Last line.\n"


def test_zero_vad_silence_still_emits_last_line(tmp_path: Path) -> None:
    wav_path = tmp_path / "end.wav"
    _write_wav(wav_path, SPEECH)
    out_path = tmp_path / "end.en.txt"

    def transcribe(_pcm: bytes, **_kwargs: object) -> str:
        return "Last line."

    cues = run_wav(
        wav_path=wav_path,
        out_path=out_path,
        transcribe=transcribe,
        chunker=_chunker(min_silence_ms=0),
        mode="ko_to_en",
        min_silence_ms=0,
    )
    assert len(cues) == 1
    assert out_path.read_text(encoding="utf-8") == "Last line.\n"


def test_empty_whisper_result_is_skipped(tmp_path: Path) -> None:
    wav_path = tmp_path / "empty.wav"
    _write_wav(wav_path, SPEECH)
    out_path = tmp_path / "empty.en.txt"

    def transcribe(_pcm: bytes, **_kwargs: object) -> str:
        return "  "

    cues = run_wav(
        wav_path=wav_path,
        out_path=out_path,
        transcribe=transcribe,
        chunker=_chunker(),
        mode="ko_to_en",
        min_silence_ms=400,
    )
    assert len(cues) == 0
    assert out_path.read_text(encoding="utf-8") == ""


def test_each_line_is_timed_from_its_speech_into_the_recording(tmp_path: Path) -> None:
    # Speech 0–0.512 s, silence to 1.472 s, speech to 1.984 s.
    wav_path = tmp_path / "sermon.wav"
    _write_wav(wav_path, np.concatenate([SPEECH, SILENCE, SPEECH]))
    lines = iter(["One.", "Two."])

    cues = run_wav(
        wav_path=wav_path,
        out_path=tmp_path / "sermon.en.txt",
        transcribe=lambda _pcm, **_kwargs: next(lines),
        chunker=_chunker(),
        mode="ko_to_en",
        min_silence_ms=400,
        start_s=0.0,
    )
    assert [c.text for c in cues] == ["One.", "Two."]
    # Each starts with its 200 ms preroll and ends where the speech stopped.
    assert cues[0].start_s == pytest.approx(0.0, abs=0.04)
    assert cues[0].end_s == pytest.approx(0.512, abs=0.04)
    assert cues[1].start_s == pytest.approx(1.472 - 0.2, abs=0.04)
    assert cues[1].end_s == pytest.approx(1.984, abs=0.04)


def test_times_count_from_the_start_of_the_recording_not_the_window(tmp_path: Path) -> None:
    wav_path = tmp_path / "sermon.wav"
    _write_wav(wav_path, np.concatenate([SILENCE, SPEECH, SILENCE]))  # speech from 0.96 s

    cues = run_wav(
        wav_path=wav_path,
        out_path=tmp_path / "sermon.en.txt",
        transcribe=lambda _pcm, **_kwargs: "Line.",
        chunker=_chunker(),
        mode="ko_to_en",
        min_silence_ms=400,
        start_s=0.5,
    )
    assert len(cues) == 1
    assert cues[0].start_s == pytest.approx(0.96 - 0.2, abs=0.04)
    assert cues[0].end_s == pytest.approx(0.96 + 0.512, abs=0.04)


SHORT = np.full(FRAME * 25, 8000, dtype=np.int16)  # 0.8 s
GAP = np.zeros(FRAME * 16, dtype=np.int16)  # 0.512 s, enough to cut at 400 ms


def test_joining_sends_two_short_pieces_as_one_line(tmp_path: Path) -> None:
    wav_path = tmp_path / "sermon.wav"
    _write_wav(wav_path, np.concatenate([SHORT, GAP, SHORT, GAP]))
    calls: list[int] = []

    def transcribe(pcm: bytes, **_kwargs: object) -> str:
        calls.append(len(pcm))
        return "Joined."

    cues = run_wav(
        wav_path=wav_path,
        out_path=tmp_path / "sermon.en.txt",
        transcribe=transcribe,
        chunker=_chunker(),
        mode="en_to_en",
        min_silence_ms=400,
        join_short=True,
    )
    assert len(calls) == 1
    assert [c.text for c in cues] == ["Joined."]
    assert cues[0].start_s == pytest.approx(0.0, abs=0.04)
    assert cues[0].end_s == pytest.approx(0.8 + 0.512 + 0.8, abs=0.08)


def test_a_short_piece_goes_alone_when_the_next_is_far(tmp_path: Path) -> None:
    wav_path = tmp_path / "sermon.wav"
    far = np.zeros(FRAME * 100, dtype=np.int16)  # 3.2 s
    _write_wav(wav_path, np.concatenate([SHORT, far, SHORT, GAP]))
    calls: list[int] = []

    def transcribe(pcm: bytes, **_kwargs: object) -> str:
        calls.append(len(pcm))
        return f"Line {len(calls)}."

    cues = run_wav(
        wav_path=wav_path,
        out_path=tmp_path / "sermon.en.txt",
        transcribe=transcribe,
        chunker=_chunker(),
        mode="en_to_en",
        min_silence_ms=400,
        join_short=True,
    )
    assert len(calls) == 2
    assert [c.text for c in cues] == ["Line 1.", "Line 2."]


def test_en_to_ko_reaches_the_engine_as_asked(tmp_path: Path) -> None:
    wav_path = tmp_path / "sermon.wav"
    _write_wav(wav_path, SPEECH)
    seen: list[dict] = []

    def transcribe(_pcm: bytes, **kwargs: object) -> str:
        seen.append(kwargs)
        return "오늘 우리는"

    cues = run_wav(
        wav_path=wav_path,
        out_path=tmp_path / "sermon.ko.txt",
        transcribe=transcribe,
        chunker=_chunker(),
        mode="en_to_ko",
        min_silence_ms=400,
    )
    assert seen[0]["language"] == "en"
    assert seen[0]["target_language"] == "ko"
    assert (tmp_path / "sermon.ko.txt").read_text(encoding="utf-8") == "오늘 우리는\n"
    assert len(cues) == 1
