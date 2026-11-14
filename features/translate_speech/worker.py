from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from core.event_bus import EventBus
from features.capture_audio.events import AudioChunkEvent
from features.operator_control.events import (
    CaptionsStateEvent,
    ConditionOnPreviousTextEvent,
    ModeChangedEvent,
    TranslateMode,
    languages_for,
    korean_captions,
)
from features.translate_speech.captioner import Captioner
from features.translate_speech.events import SubtitleEvent

logger = logging.getLogger(__name__)

TranscribeFn = Callable[..., str]


class TranslateWorker:
    def __init__(
        self,
        bus: EventBus,
        transcribe: TranscribeFn,
        *,
        mode: str = "ko_to_en",
        queue_size: int = 4,
        engine: str = "whisper",
        translate=None,
        condition_on_previous_text: bool = False,
        korean: bool = False,
    ) -> None:
        self._bus = bus
        self._captioner = Captioner(
            transcribe,
            engine=engine,
            translate=translate,
            condition_on_previous_text=condition_on_previous_text,
        )
        # The mode the booth picked, and whether `transcribe` writes Korean (Seamless on Hermeneia).
        self._picked = TranslateMode(mode)
        self._korean = korean
        self._mode = self._runnable_mode()
        self._queue: asyncio.Queue[AudioChunkEvent] = asyncio.Queue(maxsize=queue_size)
        self._active = True
        bus.subscribe(CaptionsStateEvent, self._on_captions)
        bus.subscribe(ModeChangedEvent, self._on_mode)
        bus.subscribe(ConditionOnPreviousTextEvent, self._on_previous_text)
        bus.subscribe(AudioChunkEvent, self._on_chunk)

    def use_transcribe(self, transcribe: TranscribeFn, *, korean: bool = False) -> None:
        """Speech moved (this PC, Hermeneia, another model): the next chunk goes to `transcribe`."""
        self._captioner.use_recognize(transcribe)
        self._korean = korean
        self._mode = self._runnable_mode()

    def _runnable_mode(self) -> TranslateMode:
        """Whisper cannot write Korean, so there en_to_ko falls back to the sermon mode."""
        if korean_captions(self._picked) and not self._korean:
            return TranslateMode.KO_TO_EN
        return self._picked

    def drop_waiting(self) -> None:
        """Chunks queued while Gemini ran are stale, so they never become captions."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except (asyncio.QueueEmpty, ValueError):
                break

    async def _on_previous_text(self, event: ConditionOnPreviousTextEvent) -> None:
        self._captioner.set_condition_on_previous_text(event.condition_on_previous_text)

    async def _on_captions(self, event: CaptionsStateEvent) -> None:
        self._active = event.is_active
        if not event.is_active:
            self._captioner.clear_previous()
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    break
                except ValueError:
                    break

    async def _on_mode(self, event: ModeChangedEvent) -> None:
        try:
            self._picked = TranslateMode(event.mode)
        except ValueError:
            self._picked = TranslateMode.KO_TO_EN
        self._mode = self._runnable_mode()

    async def _on_chunk(self, event: AudioChunkEvent) -> None:
        if not self._active:
            return
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                pass
            except ValueError:
                pass
        await self._queue.put(event)

    async def run(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self._process(event)
            except Exception:
                logger.exception("translate worker failed")
            finally:
                self._queue.task_done()

    async def _process(self, event: AudioChunkEvent) -> None:
        if not self._active:
            return
        text = await asyncio.to_thread(
            self._captioner.line,
            event.pcm_s16le,
            direction=self._mode.value,
        )
        if not self._active:
            return
        cleaned = (text or "").strip()
        if not cleaned:
            return
        logger.info("%s", cleaned)
        source, target = languages_for(self._mode)
        now = time.time()
        await self._bus.publish(
            SubtitleEvent(
                chunk_id=event.chunk_id,
                text=cleaned,
                source_language=source,
                target_language=target,
                mode=self._mode.value,
                created_at=now,
                duration_s=event.duration_s,
                # The chunk's start is on the monotonic clock; move it to wall time.
                spoken_at=now - (time.monotonic() - event.started_at),
            )
        )

