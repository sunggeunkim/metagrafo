from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from core.event_bus import EventBus
from features.capture_audio.events import AudioChunkEvent
from features.operator_control.events import (
    CaptionsStateEvent,
    ModeChangedEvent,
    TranslateMode,
    languages_for,
)
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
        initial_prompt: str = "",
        queue_size: int = 4,
    ) -> None:
        self._bus = bus
        self._transcribe = transcribe
        self._mode = TranslateMode(mode)
        self._prompt = initial_prompt
        self._queue: asyncio.Queue[AudioChunkEvent] = asyncio.Queue(maxsize=queue_size)
        self._active = False
        bus.subscribe(CaptionsStateEvent, self._on_captions)
        bus.subscribe(ModeChangedEvent, self._on_mode)
        bus.subscribe(AudioChunkEvent, self._on_chunk)

    async def _on_captions(self, event: CaptionsStateEvent) -> None:
        self._active = event.is_active
        if not event.is_active:
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
            self._mode = TranslateMode(event.mode)
        except ValueError:
            self._mode = TranslateMode.KO_TO_EN

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
        if self._mode is TranslateMode.EN_TO_EN:
            language, task = "en", "transcribe"
        else:
            language, task = "ko", "translate"
        text = await asyncio.to_thread(
            self._transcribe,
            event.pcm_s16le,
            language=language,
            task=task,
            initial_prompt=self._prompt,
        )
        if not self._active:
            return
        cleaned = (text or "").strip()
        if not cleaned:
            return
        source, target = languages_for(self._mode)
        await self._bus.publish(
            SubtitleEvent(
                chunk_id=event.chunk_id,
                text=cleaned,
                source_language=source,
                target_language=target,
                mode=self._mode.value,
                created_at=time.time(),
                duration_s=event.duration_s,
            )
        )
