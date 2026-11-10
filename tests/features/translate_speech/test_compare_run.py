from __future__ import annotations

import asyncio
import wave
from pathlib import Path
from types import SimpleNamespace

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


async def _cleaned(line: str, _shown: list[str]) -> str:
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
