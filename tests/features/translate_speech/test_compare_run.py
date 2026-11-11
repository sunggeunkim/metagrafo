from __future__ import annotations

import asyncio
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from features.translate_speech.compare_run import compare_wav

BYTES_PER_S = 32000


def _write_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * int(16000 * seconds))


def _heard(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        go_away=None,
        server_content=SimpleNamespace(
            input_transcription=SimpleNamespace(text=text),
            output_transcription=None,
            turn_complete=False,
        ),
    )


def _said(text: str, *, turn_complete: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        go_away=None,
        server_content=SimpleNamespace(
            input_transcription=None,
            output_transcription=SimpleNamespace(text=text),
            turn_complete=turn_complete,
        ),
    )


class _ScriptedSocket:
    """Plays the Gemini live socket: each response comes once that much audio has been sent."""

    def __init__(self, script: list[tuple[float, SimpleNamespace]]) -> None:
        self._script = script
        self._next = 0
        self.sent_bytes = 0

    async def __aenter__(self) -> _ScriptedSocket:
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    def _due(self) -> bool:
        return (
            self._next < len(self._script)
            and self._script[self._next][0] * BYTES_PER_S <= self.sent_bytes
        )

    async def send_realtime_input(self, *, audio: object) -> None:
        self.sent_bytes += len(audio.data)
        # The next chunk goes only after every response due by now has been handled.
        while self._due():
            await asyncio.sleep(0)

    async def receive(self):
        while self._next < len(self._script):
            at_s, response = self._script[self._next]
            while self.sent_bytes < at_s * BYTES_PER_S:
                await asyncio.sleep(0)
            yield response
            self._next += 1
        await asyncio.Event().wait()
        yield None


async def _no_wait(_position_s: float) -> None:
    # Lets the lines being cleaned or translated finish before the next 100 ms of audio.
    for _ in range(20):
        await asyncio.sleep(0)


def _rows(out: Path) -> list[str]:
    return [line for line in out.read_text(encoding="utf-8").splitlines() if line.startswith("| ")][2:]


async def _cleaned(line: str, _shown: list[str], _said: str) -> str:
    return line.upper()


async def test_one_sermon_line_is_written_both_ways_with_its_time_and_delay(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, 4.0)
    out = tmp_path / "sermon.compare.md"
    socket = _ScriptedSocket(
        [
            (1.0, _heard("오늘은 기도로")),
            (1.5, _heard("시작하겠습니다.")),
            (2.0, _said("Today we begin")),
            (2.5, _said("with prayer.")),
        ]
    )

    async def translate(korean: str, _previous: list[str]) -> str:
        assert korean == "오늘은 기도로 시작하겠습니다."
        return "Let us begin today with prayer."

    await compare_wav(
        wav_path=wav,
        out_path=out,
        connect=lambda _mode: socket,
        cleanup=_cleaned,
        translate=translate,
        pace=_no_wait,
        cleanup_timeout_s=0.05,
    )
    assert _rows(out) == [
        "| 0:01.5 | | Let us begin today with prayer. (+0.5 s) | 오늘은 기도로 시작하겠습니다. |",
        "| 0:02.5 | TODAY WE BEGIN WITH PRAYER. (+0.5 s) | | |",
    ]


async def test_c_closes_a_korean_sentence_at_a_pause_and_gives_the_model_its_last_three_lines(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, 12.0)
    out = tmp_path / "sermon.compare.md"
    socket = _ScriptedSocket(
        [
            (1.0, _heard("첫째.")),
            (2.0, _heard("둘째.")),
            (3.0, _heard("셋째.")),
            # No end mark: the 1.2 s pause after it closes the sentence.
            (4.0, _heard("넷째 그리고")),
            (4.5, _heard("다섯째")),
        ]
    )
    asked: list[tuple[str, list[str]]] = []

    async def translate(korean: str, previous: list[str]) -> str:
        asked.append((korean, previous))
        return f"Line {len(asked)}."

    await compare_wav(
        wav_path=wav,
        out_path=out,
        connect=lambda _mode: socket,
        cleanup=_cleaned,
        translate=translate,
        pace=_no_wait,
        cleanup_timeout_s=0.05,
    )
    assert asked[-1] == ("넷째 그리고 다섯째", ["Line 1.", "Line 2.", "Line 3."])
    assert _rows(out)[-1] == "| 0:05.7 | | Line 4. (+1.7 s) | 넷째 그리고 다섯째 |"


async def test_a_failed_c_translation_is_noted_without_the_key_and_the_run_goes_on(tmp_path: Path, caplog) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, 4.0)
    out = tmp_path / "sermon.compare.md"
    socket = _ScriptedSocket([(1.0, _heard("첫째.")), (2.0, _heard("둘째."))])

    async def translate(korean: str, _previous: list[str]) -> str:
        if korean == "첫째.":
            raise RuntimeError("429 quota exhausted for key secret-key-7")
        return "Second."

    with caplog.at_level("WARNING"):
        await asyncio.wait_for(
            compare_wav(
                wav_path=wav,
                out_path=out,
                connect=lambda _mode: socket,
                cleanup=_cleaned,
                translate=translate,
                api_key="secret-key-7",
                pace=_no_wait,
                cleanup_timeout_s=0.05,
            ),
            timeout=10,
        )
    assert _rows(out) == [
        "| 0:01.0 | | (translation failed: 429 quota exhausted for key [key]) | 첫째. |",
        "| 0:02.0 | | Second. (+0.0 s) | 둘째. |",
    ]
    assert "secret-key-7" not in out.read_text(encoding="utf-8")
    assert "secret-key-7" not in caplog.text
    assert "translation failed: 429 quota exhausted for key [key]" in caplog.text


async def test_the_run_reconnects_when_the_socket_goes_away_and_carries_on_from_the_same_audio(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, 6.0)
    out = tmp_path / "sermon.compare.md"
    going_away = SimpleNamespace(go_away=SimpleNamespace(time_left="0s"), server_content=None)
    sockets = [
        _ScriptedSocket([(1.0, _said("The first line of the sermon today.")), (2.0, going_away)]),
        # This socket's times count from its own first chunk. go_away came while the chunk
        # ending at 2.0 s was still being sent, so that chunk goes again: it starts at 1.9 s.
        _ScriptedSocket([(1.0, _said("The second line of the sermon today."))]),
    ]
    opened: list[_ScriptedSocket] = []

    def connect(_mode: object) -> _ScriptedSocket:
        opened.append(sockets[len(opened)])
        return opened[-1]

    await compare_wav(
        wav_path=wav,
        out_path=out,
        connect=connect,
        cleanup=_cleaned,
        translate=_cleaned,
        pace=_no_wait,
        cleanup_timeout_s=0.05,
    )
    assert _rows(out) == [
        "| 0:01.0 | THE FIRST LINE OF THE SERMON TODAY. (+0.0 s) | | |",
        "| 0:02.9 | THE SECOND LINE OF THE SERMON TODAY. (+0.0 s) | | |",
    ]
    assert len(opened) == 2


async def test_a_socket_that_fails_is_retried_and_logged_without_the_key(tmp_path: Path, caplog) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, 3.0)
    out = tmp_path / "sermon.compare.md"
    working = _ScriptedSocket([(1.0, _said("The first line of the sermon today."))])
    attempts: list[int] = []

    def connect(_mode: object) -> _ScriptedSocket:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("503 unavailable for key secret-key-7")
        return working

    with caplog.at_level("WARNING"):
        await compare_wav(
            wav_path=wav,
            out_path=out,
            connect=connect,
            cleanup=_cleaned,
            translate=_cleaned,
            api_key="secret-key-7",
            pace=_no_wait,
            cleanup_timeout_s=0.05,
            retry_wait_s=0.0,
        )
    assert _rows(out) == ["| 0:01.0 | THE FIRST LINE OF THE SERMON TODAY. (+0.0 s) | | |"]
    assert "socket failed, reconnecting: 503 unavailable for key [key]" in caplog.text
    assert "secret-key-7" not in caplog.text


async def test_a_socket_that_keeps_failing_ends_the_run(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, 3.0)

    def connect(_mode: object) -> _ScriptedSocket:
        raise RuntimeError("401 bad key")

    with pytest.raises(RuntimeError, match="401 bad key"):
        await asyncio.wait_for(
            compare_wav(
                wav_path=wav,
                out_path=tmp_path / "sermon.compare.md",
                connect=connect,
                cleanup=_cleaned,
                translate=_cleaned,
                pace=_no_wait,
                cleanup_timeout_s=0.05,
                retry_wait_s=0.0,
            ),
            timeout=10,
        )
    assert (tmp_path / "sermon.compare.md").exists()


class _SendFailsSocket(_ScriptedSocket):
    """Takes `ok` chunks, then fails to send the next one."""

    def __init__(self, ok: int) -> None:
        super().__init__([])
        self._ok = ok

    async def send_realtime_input(self, *, audio: object) -> None:
        if self._ok == 0:
            raise RuntimeError("socket closed")
        self._ok -= 1
        await super().send_realtime_input(audio=audio)


async def _run_on(sockets: list[_ScriptedSocket], wav: Path, out: Path) -> None:
    opened: list[_ScriptedSocket] = []

    def connect(_mode: object) -> _ScriptedSocket:
        opened.append(sockets[len(opened)])
        return opened[-1]

    await compare_wav(
        wav_path=wav,
        out_path=out,
        connect=connect,
        cleanup=_cleaned,
        translate=_cleaned,
        pace=_no_wait,
        cleanup_timeout_s=0.05,
        retry_wait_s=0.0,
    )


async def test_a_chunk_that_fails_to_send_is_sent_again_on_the_next_socket(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, 2.0)
    healthy = _ScriptedSocket([])
    await _run_on([healthy], wav, tmp_path / "healthy.md")
    failing, retried = _SendFailsSocket(ok=5), _ScriptedSocket([])
    await _run_on([failing, retried], wav, tmp_path / "retried.md")
    assert failing.sent_bytes + retried.sent_bytes == healthy.sent_bytes


async def test_sends_that_keep_failing_end_the_run(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, 2.0)
    with pytest.raises(RuntimeError, match="socket closed"):
        await asyncio.wait_for(
            _run_on([_SendFailsSocket(ok=0) for _ in range(3)], wav, tmp_path / "out.md"), timeout=10
        )


class _LateSocket(_ScriptedSocket):
    """Says its last line only after the audio has stopped coming."""

    def __init__(self) -> None:
        super().__init__([])

    async def receive(self):
        while True:
            before = self.sent_bytes
            await asyncio.sleep(0.05)
            if self.sent_bytes == before:
                break
        yield _said("The last line of the sermon for today.", turn_complete=True)
        await asyncio.Event().wait()
        yield None


async def test_a_line_gemini_says_after_the_last_audio_is_still_written(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, 1.0)
    out = tmp_path / "sermon.compare.md"
    await compare_wav(
        wav_path=wav,
        out_path=out,
        connect=lambda _mode: _LateSocket(),
        cleanup=_cleaned,
        translate=_cleaned,
        pace=_no_wait,
        cleanup_timeout_s=0.05,
        drain_s=1.0,
    )
    assert [row.split(" | ")[1] for row in _rows(out)] == ["THE LAST LINE OF THE SERMON FOR TODAY. (+0.0 s)"]


class _GoesAwayMidSendSocket(_ScriptedSocket):
    """Takes `ok` chunks, then goes away while the next one is still being sent."""

    def __init__(self, ok: int) -> None:
        super().__init__([])
        self._ok = ok
        self._stuck = asyncio.Event()

    async def send_realtime_input(self, *, audio: object) -> None:
        if self._ok == 0:
            self._stuck.set()
            await asyncio.Event().wait()
        self._ok -= 1
        await super().send_realtime_input(audio=audio)

    async def receive(self):
        await self._stuck.wait()
        yield SimpleNamespace(go_away=SimpleNamespace(time_left="0s"), server_content=None)


async def test_a_chunk_cut_off_by_the_socket_going_away_is_sent_again(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, 2.0)
    healthy = _ScriptedSocket([])
    await _run_on([healthy], wav, tmp_path / "healthy.md")
    leaving, next_socket = _GoesAwayMidSendSocket(ok=5), _ScriptedSocket([])
    await _run_on([leaving, next_socket], wav, tmp_path / "retried.md")
    assert leaving.sent_bytes + next_socket.sent_bytes == healthy.sent_bytes


class _ClosesAtOnceSocket(_ScriptedSocket):
    """Goes away before taking any audio."""

    def __init__(self) -> None:
        super().__init__([])

    async def send_realtime_input(self, *, audio: object) -> None:
        await asyncio.Event().wait()

    async def receive(self):
        yield SimpleNamespace(go_away=SimpleNamespace(time_left="0s"), server_content=None)


async def test_sockets_that_keep_closing_before_any_audio_end_the_run(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, 2.0)
    with pytest.raises(RuntimeError, match="closed 3 times without taking any audio"):
        await asyncio.wait_for(
            _run_on([_ClosesAtOnceSocket() for _ in range(3)], wav, tmp_path / "out.md"), timeout=10
        )
    assert (tmp_path / "out.md").exists()


async def test_c_splits_a_fragment_that_holds_more_than_one_korean_sentence(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, 4.0)
    out = tmp_path / "sermon.compare.md"
    socket = _ScriptedSocket([(1.0, _heard("첫째입니다. 둘째입니까? 셋째")), (1.5, _heard("입니다."))])
    asked: list[str] = []

    async def translate(korean: str, _previous: list[str]) -> str:
        asked.append(korean)
        return "Line."

    await compare_wav(
        wav_path=wav,
        out_path=out,
        connect=lambda _mode: socket,
        cleanup=_cleaned,
        translate=translate,
        pace=_no_wait,
        cleanup_timeout_s=0.05,
    )
    assert asked == ["첫째입니다.", "둘째입니까?", "셋째 입니다."]


async def test_a_line_finished_after_the_audio_ran_out_is_timed_when_it_finished(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, 1.0)
    out = tmp_path / "sermon.compare.md"
    socket = _ScriptedSocket([(0.5, _heard("첫째."))])

    async def slow_translate(_korean: str, _previous: list[str]) -> str:
        # The audio and its tail are all sent within this wait.
        await asyncio.sleep(0.6)
        return "First."

    await compare_wav(
        wav_path=wav,
        out_path=out,
        connect=lambda _mode: socket,
        cleanup=_cleaned,
        translate=slow_translate,
        pace=_no_wait,
        cleanup_timeout_s=0.05,
        tail_s=0.0,
        drain_s=0.0,
    )
    [row] = _rows(out)
    minutes, seconds = row.split(" | ")[0].removeprefix("| ").split(":")
    at_s = int(minutes) * 60 + float(seconds)
    # The audio ends near 1.0 s; the line finished about 0.6 s of work after the sentence was heard at 0.5 s.
    assert 1.4 <= at_s <= 2.5


async def test_b_cleanup_gets_what_the_speaker_said(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, 4.0)
    socket = _ScriptedSocket(
        [
            (1.0, _heard("오늘은 기도로")),
            # Like real input_transcription fragments, each after the first starts with a space.
            (1.5, _heard(" 시작하겠습니다.")),
            (2.0, _said("Today we begin")),
            (2.5, _said("with prayer.")),
        ]
    )
    asked: list[str] = []

    async def cleanup(line: str, _shown: list[str], said: str) -> str:
        asked.append(said)
        return line

    await compare_wav(
        wav_path=wav,
        out_path=tmp_path / "sermon.compare.md",
        connect=lambda _mode: socket,
        cleanup=cleanup,
        translate=_cleaned,
        pace=_no_wait,
        cleanup_timeout_s=0.05,
    )
    assert asked == ["오늘은 기도로 시작하겠습니다."]
