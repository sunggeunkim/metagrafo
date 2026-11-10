"""Prototype for #41: caption one recording two ways and write them side by side.

B is what the overlay shows today: Gemini live with Whole sentences and Clean up wording on.
C translates each Korean sentence Gemini live heard, in one text-model call.
One socket pass feeds both. Nothing here is wired into the live pipeline or booth control.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from core.event_bus import EventBus
from features.capture_audio.wav import frames_from_wav
from features.operator_control.events import TranslateMode
from features.translate_speech.events import SubtitleEvent
from features.translate_speech.gemini_live import (
    CLEANUP_TIMEOUT_S,
    Cleanup,
    Connect,
    GeminiLiveTranslator,
    LiveConnection,
    _pcm_blob,
    _redact,
)
from features.translate_speech.live_line import LiveLine

logger = logging.getLogger(__name__)

# A Korean sentence and the English lines C wrote before it in; its English out.
Translate = Callable[[str, list[str]], Awaitable[str]]
# Called after each chunk with the seconds of audio sent so far; returns when the next may go.
Pace = Callable[[float], Awaitable[None]]

BYTES_PER_S = 32000  # 16 kHz mono s16le
CHUNK_BYTES = 3200  # 100 ms, like the booth's program audio
TAIL_S = 5.0
CONTEXT_LINES = 3


@dataclass
class _Row:
    at_s: float
    b: str = ""
    c: str = ""
    korean: str = ""


def real_time() -> Pace:
    """Send audio as fast as it was spoken, as the booth does."""
    started: float | None = None

    async def pace(sent_s: float) -> None:
        nonlocal started
        now = asyncio.get_running_loop().time()
        if started is None:
            started = now - sent_s
        await asyncio.sleep(max(0.0, started + sent_s - now))

    return pace


async def compare_wav(
    *,
    wav_path: Path,
    out_path: Path,
    connect: Connect,
    cleanup: Cleanup,
    translate: Translate,
    api_key: str = "",
    start_s: float = 0.0,
    end_s: float | None = None,
    pace: Pace | None = None,
    quiet_s: float = 1.2,
    cleanup_timeout_s: float = CLEANUP_TIMEOUT_S,
    tail_s: float = TAIL_S,
) -> Path:
    pcm = b"".join(frames_from_wav(wav_path, min_silence_ms=0, start_s=start_s, end_s=end_s))
    # Silence after the recording lets the socket finish the last lines.
    pcm += b"\x00" * (int(tail_s * BYTES_PER_S) // CHUNK_BYTES * CHUNK_BYTES)
    run = _CompareRun(
        pcm,
        connect=connect,
        cleanup=cleanup,
        translate=translate,
        api_key=api_key,
        start_s=start_s,
        pace=pace or real_time(),
        quiet_s=quiet_s,
        cleanup_timeout_s=cleanup_timeout_s,
    )
    await run.go()
    out_path = Path(out_path)
    out_path.write_text(_markdown(Path(wav_path).name, run.rows), encoding="utf-8")
    return out_path


class _CompareRun:
    def __init__(
        self,
        pcm: bytes,
        *,
        connect: Connect,
        cleanup: Cleanup,
        translate: Translate,
        api_key: str,
        start_s: float,
        pace: Pace,
        quiet_s: float,
        cleanup_timeout_s: float,
    ) -> None:
        self._pcm = pcm
        self._sent = 0
        self._connect = connect
        self._translate = translate
        self._api_key = api_key
        self._start_s = start_s
        self._pace = pace
        self._quiet_s = quiet_s
        self._cleanup_timeout_s = cleanup_timeout_s
        self.rows: list[_Row] = []
        bus = EventBus()
        bus.subscribe(SubtitleEvent, self._on_b_line)
        self._b = GeminiLiveTranslator(
            bus,
            connect=connect,
            api_key=api_key,
            clock=self._now,
            whole_sentences=True,
            quiet_s=quiet_s,
            clean_up_wording=True,
            cleanup=cleanup,
            cleanup_timeout_s=cleanup_timeout_s,
        )
        self._korean = LiveLine()
        self._last_heard_s = 0.0
        self._sentences: asyncio.Queue[tuple[str, float | None]] = asyncio.Queue()
        self._english: list[str] = []

    def _now(self) -> float:
        """Seconds into the recording: the audio sent so far, which real-time pacing keeps on the wall clock."""
        return self._start_s + self._sent / BYTES_PER_S

    async def go(self) -> None:
        translator = asyncio.create_task(self._translate_in_order(), name="compare-c")
        try:
            while self._sent < len(self._pcm):
                async with self._connect(TranslateMode.KO_TO_EN) as session:
                    await self._pump(session)
                await self._b.apply_transcript("", turn_complete=True)
            self._close_korean()
            await self._sentences.join()
            # Each B line still being cleaned shows by its cleanup timeout.
            await asyncio.sleep(self._cleanup_timeout_s + 0.1)
        finally:
            translator.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await translator

    async def _pump(self, session: LiveConnection) -> None:
        sender = asyncio.create_task(self._send(session), name="compare-send")
        receiver = asyncio.create_task(self._receive(session), name="compare-recv")
        try:
            done, _ = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in (sender, receiver):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def _send(self, session: LiveConnection) -> None:
        while self._sent < len(self._pcm):
            chunk = self._pcm[self._sent : self._sent + CHUNK_BYTES]
            self._sent += len(chunk)
            await session.send_realtime_input(audio=_pcm_blob(chunk))
            if self._korean.text and self._now() - self._last_heard_s >= self._quiet_s:
                self._close_korean()
            await self._pace(self._sent / BYTES_PER_S)

    async def _receive(self, session: LiveConnection) -> None:
        async for response in session.receive():
            content = getattr(response, "server_content", None)
            if content is not None:
                heard = getattr(content, "input_transcription", None)
                heard_text = getattr(heard, "text", None) if heard is not None else None
                if heard_text and heard_text.strip():
                    self._hear(heard_text)
                said = getattr(content, "output_transcription", None)
                fragment = getattr(said, "text", None) if said is not None else None
                turn_complete = bool(getattr(content, "turn_complete", False))
                if fragment or turn_complete:
                    await self._b.apply_transcript(fragment or "", turn_complete=turn_complete)
            if getattr(response, "go_away", None) is not None:
                return

    def _hear(self, text: str) -> None:
        now = self._now()
        self._korean.extend(text, at=now)
        self._last_heard_s = now
        if self._korean.closes_sentence():
            self._close_korean()

    def _close_korean(self) -> None:
        heard_at = self._korean.opened_at
        sentence = self._korean.commit()
        if sentence:
            self._sentences.put_nowait((sentence, heard_at))

    async def _translate_in_order(self) -> None:
        while True:
            korean, heard_at = await self._sentences.get()
            try:
                try:
                    english = (await self._translate(korean, self._english[-CONTEXT_LINES:])).strip()
                except Exception as exc:
                    # A failed line is noted where it would have been; the run goes on.
                    reason = _redact(exc, self._api_key)
                    logger.warning("translation failed: %s", reason)
                    self.rows.append(_Row(self._now(), c=f"(translation failed: {reason})", korean=korean))
                    continue
                self._english.append(english)
                now = self._now()
                self.rows.append(_Row(now, c=f"{english}{_delay(now, heard_at)}", korean=korean))
            finally:
                self._sentences.task_done()

    async def _on_b_line(self, event: SubtitleEvent) -> None:
        if event.provisional:
            return
        self.rows.append(_Row(event.created_at, b=f"{event.text}{_delay(event.created_at, event.spoken_at)}"))


def _delay(now: float, heard_at: float | None) -> str:
    return "" if heard_at is None else f" (+{now - heard_at:.1f} s)"


def _time(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}:{rest:04.1f}"


def _cell(text: str) -> str:
    return " " if not text else f" {text.replace('|', '\\|')} "


def _markdown(name: str, rows: list[_Row]) -> str:
    lines = [
        f"# {name}: B vs C",
        "",
        "- B: Gemini live with Whole sentences and Clean up wording on (what the overlay shows).",
        "- C: each Korean sentence Gemini live heard, translated and cleaned in one text-model call.",
        "- Time is from the start of the recording. (+N s) is the delay from the line's first heard word.",
        "",
        "| Time | B: live + cleanup | C: sentence translation | C: Korean |",
        "| --- | --- | --- | --- |",
    ]
    for row in sorted(rows, key=lambda r: r.at_s):
        lines.append("|" + "|".join(_cell(t) for t in (_time(row.at_s), row.b, row.c, row.korean)) + "|")
    return "\n".join(lines) + "\n"
