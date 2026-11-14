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
        seamless: bool = False,
        join_under_s: float = 2.0,
        hold_s: float = 1.5,
    ) -> None:
        self._bus = bus
        self._captioner = Captioner(
            transcribe,
            engine=engine,
            translate=translate,
            condition_on_previous_text=condition_on_previous_text,
        )
        # The mode the booth picked, and whether `transcribe` is Seamless on Hermeneia, which
        # writes Korean, and makes text up from a second of audio, so short pieces are joined.
        self._picked = TranslateMode(mode)
        self._seamless = seamless
        self._mode = self._runnable_mode()
        self._join_under_s = join_under_s
        self._hold_s = hold_s
        self._held: list[AudioChunkEvent] = []
        self._send_held: asyncio.TimerHandle | None = None
        self._queue: asyncio.Queue[AudioChunkEvent] = asyncio.Queue(maxsize=queue_size)
        self._active = True
        bus.subscribe(CaptionsStateEvent, self._on_captions)
        bus.subscribe(ModeChangedEvent, self._on_mode)
        bus.subscribe(ConditionOnPreviousTextEvent, self._on_previous_text)
        bus.subscribe(AudioChunkEvent, self._on_chunk)

    def use_transcribe(self, transcribe: TranscribeFn, *, seamless: bool = False) -> None:
        """Speech moved (this PC, Hermeneia, another model): the next chunk goes to `transcribe`."""
        self._captioner.use_recognize(transcribe)
        self._seamless = seamless
        self._mode = self._runnable_mode()
        self._drop_held()

    def _runnable_mode(self) -> TranslateMode:
        """Whisper cannot write Korean, so there en_to_ko falls back to the sermon mode."""
        if korean_captions(self._picked) and not self._seamless:
            return TranslateMode.KO_TO_EN
        return self._picked

    def drop_waiting(self) -> None:
        """Chunks queued while Gemini ran are stale, so they never become captions."""
        self._drop_held()
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
            self._drop_held()
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
        if self._seamless:
            self._held.append(event)
            if sum(piece.duration_s for piece in self._held) < self._join_under_s:
                # Waits for the next piece; sent alone if the speaker stops here.
                if self._send_held is None:
                    self._send_held = asyncio.get_running_loop().call_later(self._hold_s, self._send_held_now)
                return
            event = self._take_held()
        self._enqueue(event)

    def _send_held_now(self) -> None:
        self._send_held = None
        if self._held and self._active:
            self._enqueue(self._take_held())

    def _take_held(self) -> AudioChunkEvent:
        """The held pieces as one, timed from the first."""
        held, self._held = self._held, []
        if self._send_held is not None:
            self._send_held.cancel()
            self._send_held = None
        if len(held) == 1:
            return held[0]
        return AudioChunkEvent(
            chunk_id=held[0].chunk_id,
            pcm_s16le=b"".join(piece.pcm_s16le for piece in held),
            sample_rate=held[0].sample_rate,
            started_at=held[0].started_at,
            duration_s=sum(piece.duration_s for piece in held),
        )

    def _drop_held(self) -> None:
        self._held = []
        if self._send_held is not None:
            self._send_held.cancel()
            self._send_held = None

    def _enqueue(self, event: AudioChunkEvent) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                pass
            except ValueError:
                pass
        self._queue.put_nowait(event)

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

