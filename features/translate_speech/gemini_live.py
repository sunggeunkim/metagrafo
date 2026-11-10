from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from types import TracebackType

from core.event_bus import EventBus
from features.capture_audio.events import ProgramAudioEvent
from features.operator_control.events import (
    CaptionsStateEvent,
    ModeChangedEvent,
    TranslateMode,
    WholeSentencesEvent,
    languages_for,
)
from features.translate_speech.events import SubtitleEvent
from features.translate_speech.live_line import LiveLine
from features.translate_speech.tidy import tidy

logger = logging.getLogger(__name__)

MODEL = "gemini-3.5-live-translate-preview"
Connect = Callable[[TranslateMode], "LiveConnection"]
SHORT_SENTENCE_CHARS = 25
# Two 1920-wide overlay rows at the default 2.5 vw hold about 140 characters. The cut leaves
# slack because the overlay can wrap a full two rows onto a third (issue #36).
LONG_LINE_CHARS = 120


def echoes_target_language(mode: TranslateMode) -> bool:
    """Guest mode is already English, so the socket must echo it instead of staying silent."""
    return mode is TranslateMode.EN_TO_EN


class LiveConnection:
    """The slice of a Gemini live socket this translator uses."""

    async def __aenter__(self) -> LiveConnection:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False

    def send_realtime_input(self, **kwargs: object) -> object:
        raise NotImplementedError

    def receive(self) -> AsyncIterator[object]:
        raise NotImplementedError


def gemini_connect(api_key: str) -> Connect:
    def connect(mode: TranslateMode) -> LiveConnection:
        from google import genai
        from google.genai import types

        # google-genai 3.0 defaults to an API version that 404s the Live preview model.
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            translation_config=types.TranslationConfig(
                target_language_code=languages_for(mode)[1],
                echo_target_language=echoes_target_language(mode),
            ),
        )
        return client.aio.live.connect(model=MODEL, config=config)

    return connect


class GeminiLiveTranslator:
    def __init__(
        self,
        bus: EventBus,
        *,
        connect: Connect,
        api_key: str = "",
        mode: str = "ko_to_en",
        clock: Callable[[], float] = time.time,
        whole_sentences: bool = False,
        quiet_s: float = 1.2,
    ) -> None:
        self._bus = bus
        # The open line keeps the setting it started with; the booth's choice waits for the next.
        self._whole_sentences = whole_sentences
        self._next_whole_sentences = whole_sentences
        self._quiet_s = quiet_s
        self._quiet: asyncio.Task[None] | None = None
        self._clock = clock
        self._connect = connect
        self._api_key = api_key
        self._mode = TranslateMode(mode)
        self._line = LiveLine()
        self._audio: asyncio.Queue[bytes] = asyncio.Queue(maxsize=10)
        self._active = True
        self._stop = asyncio.Event()
        self._resumed = asyncio.Event()
        self._reconnect = asyncio.Event()
        bus.subscribe(CaptionsStateEvent, self._on_captions)
        bus.subscribe(ModeChangedEvent, self._on_mode)
        bus.subscribe(ProgramAudioEvent, self._on_audio)
        bus.subscribe(WholeSentencesEvent, self._on_whole_sentences)

    async def _on_whole_sentences(self, event: WholeSentencesEvent) -> None:
        self._next_whole_sentences = event.whole_sentences

    def use_key(self, api_key: str, connect: Connect) -> None:
        self._api_key = api_key
        self._connect = connect
        self._reconnect.set()

    async def _on_mode(self, event: ModeChangedEvent) -> None:
        try:
            mode = TranslateMode(event.mode)
        except ValueError:
            mode = TranslateMode.KO_TO_EN
        if mode == self._mode:
            return
        await self._close_open_line()
        self._mode = mode
        self._reconnect.set()

    async def _on_captions(self, event: CaptionsStateEvent) -> None:
        self._active = event.is_active
        if event.is_active:
            self._stop.clear()
            self._resumed.set()
            return
        self._resumed.clear()
        self._stop.set()
        self._drain_audio()
        self._stop_quiet_wait()
        self._line.commit()

    async def _on_audio(self, event: ProgramAudioEvent) -> None:
        if not self._active:
            return
        if self._audio.full():
            try:
                self._audio.get_nowait()
            except asyncio.QueueEmpty:
                return
            logger.warning("gemini live audio queue full; dropped the oldest chunk")
        self._audio.put_nowait(event.pcm_s16le)

    def _drain_audio(self) -> None:
        while not self._audio.empty():
            try:
                self._audio.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _close_open_line(self) -> None:
        self._stop_quiet_wait()
        spoken_at = self._line.opened_at
        finished = self._finish(self._line.commit())
        if finished and self._active:
            await self._publish(finished, provisional=False, spoken_at=spoken_at)

    async def apply_transcript(self, fragment: str, *, turn_complete: bool = False) -> None:
        if not self._active:
            return
        if not self._line.text:
            self._whole_sentences = self._next_whole_sentences
        # The first English fragment is the closest this socket gets to when the line was spoken.
        at = self._clock() if fragment.strip() else None
        shown = self._line.extend(fragment, at=at) if fragment.strip() else ""
        if self._whole_sentences:
            await self._cut_long_line(at)
        spoken_at = self._line.opened_at
        text = self._finish(self._line.text) or ""
        if (turn_complete or _ends_sentence(text)) and not self._holds_short_sentence(text):
            self._stop_quiet_wait()
            self._line.commit()
            if text:
                await self._publish(text, provisional=False, spoken_at=spoken_at)
            return
        if self._whole_sentences:
            if self._line.text:
                self._commit_when_quiet()
        elif shown:
            await self._publish(shown, provisional=True, spoken_at=spoken_at)

    def _commit_when_quiet(self) -> None:
        """Each fragment restarts the wait; a held line shows once the speaker pauses."""
        self._stop_quiet_wait()
        self._quiet = asyncio.create_task(self._commit_after_quiet(), name="gemini-live-quiet")

    def _stop_quiet_wait(self) -> None:
        """A committed or cleared line takes its quiet wait with it, so it can't cut the next line."""
        if self._quiet is not None:
            self._quiet.cancel()
            self._quiet = None

    async def _commit_after_quiet(self) -> None:
        await asyncio.sleep(self._quiet_s)
        self._quiet = None
        await self._close_open_line()

    def _finish(self, text: str | None) -> str | None:
        """The words a committed line shows: whole sentences drops fillers."""
        if text is None or not self._whole_sentences:
            return text
        return tidy(text) or None

    async def _cut_long_line(self, at: float | None) -> None:
        """Publish the front of a line that would overflow the overlay; the rest opens the next line."""
        while len(text := tidy(self._line.text)) > LONG_LINE_CHARS:
            cut = _cut_point(text, LONG_LINE_CHARS)
            if cut is None:
                return
            spoken_at = self._line.opened_at
            self._line.commit()
            await self._publish(text[:cut].strip(), provisional=False, spoken_at=spoken_at)
            if not self._active:
                # Captions went off while the front was shown; the off switch already cleared the line.
                return
            self._line.extend(text[cut:], at=at)

    def _holds_short_sentence(self, text: str) -> bool:
        """A sentence like "Yes." waits to join the next one instead of filling a line alone."""
        return self._whole_sentences and len(text) < SHORT_SENTENCE_CHARS

    async def _publish(
        self, text: str, *, provisional: bool, spoken_at: float | None = None
    ) -> None:
        if not self._active and provisional:
            return
        source, target = languages_for(self._mode)
        await self._bus.publish(
            SubtitleEvent(
                chunk_id=str(uuid.uuid4()),
                text=text,
                source_language=source,
                target_language=target,
                mode=self._mode.value,
                created_at=self._clock(),
                duration_s=0.0,
                provisional=provisional,
                spoken_at=spoken_at,
            )
        )

    async def run(self) -> None:
        while True:
            if not self._active:
                await self._resumed.wait()
                continue
            self._reconnect.clear()
            try:
                async with self._connect(self._mode) as session:
                    await self._pump(session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("gemini live session failed: %s", _redact(exc, self._api_key))
                await self._close_open_line()
                await asyncio.sleep(1)
                continue
            await self._close_open_line()
            if not self._active:
                continue

    async def _pump(self, session: LiveConnection) -> None:
        sender = asyncio.create_task(self._send(session), name="gemini-live-send")
        receiver = asyncio.create_task(self._receive(session), name="gemini-live-recv")
        stop = asyncio.create_task(self._stop.wait(), name="gemini-live-stop")
        reconnect = asyncio.create_task(self._reconnect.wait(), name="gemini-live-reconnect")
        tasks = (sender, receiver, stop, reconnect)
        try:
            await asyncio.wait(set(tasks), return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def _send(self, session: LiveConnection) -> None:
        while self._active and not self._stop.is_set():
            try:
                pcm = await asyncio.wait_for(self._audio.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            await session.send_realtime_input(
                audio=_pcm_blob(pcm),
            )

    async def _receive(self, session: LiveConnection) -> None:
        async for response in session.receive():
            content = getattr(response, "server_content", None)
            if content is not None:
                heard = getattr(content, "input_transcription", None)
                heard_text = getattr(heard, "text", None) if heard is not None else None
                if heard_text:
                    print(f"IN: {heard_text}", flush=True)
                transcription = getattr(content, "output_transcription", None)
                fragment = getattr(transcription, "text", None) if transcription is not None else None
                if fragment:
                    print(f"OUT: {fragment}", flush=True)
                turn_complete = bool(getattr(content, "turn_complete", False))
                if fragment or turn_complete:
                    await self.apply_transcript(fragment or "", turn_complete=turn_complete)
            if getattr(response, "go_away", None) is not None or not self._active:
                return


def _ends_sentence(text: str) -> bool:
    return bool(text) and text[-1] in ".?!"


def _cut_point(text: str, limit: int) -> int | None:
    """Where to end a line of at most `limit` characters: its last comma, else its last space.

    A comma in the first half would leave a stub of a line, so the space wins there.
    """
    comma = text.rfind(",", 0, limit)
    if comma >= limit // 2:
        return comma + 1
    space = text.rfind(" ", 0, limit + 1)
    return space if space > 0 else None


def _pcm_blob(pcm: bytes) -> object:
    from google.genai import types

    return types.Blob(data=pcm, mime_type="audio/pcm;rate=16000")


def _redact(exc: BaseException, api_key: str) -> str:
    text = str(exc)
    if api_key:
        text = text.replace(api_key, "[key]")
    return text
