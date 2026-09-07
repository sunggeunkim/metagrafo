from __future__ import annotations

import time
import uuid

from core.event_bus import EventBus
from features.capture_audio.chunker import SpeechChunker
from features.capture_audio.events import AudioChunkEvent
from features.operator_control.events import CaptionsStateEvent


class CaptureGate:
    def __init__(
        self,
        *,
        bus: EventBus,
        chunker: SpeechChunker,
        sample_rate: int = 16000,
    ) -> None:
        self._bus = bus
        self._chunker = chunker
        self._sample_rate = sample_rate
        self._active = False
        bus.subscribe(CaptionsStateEvent, self._on_captions)

    async def _on_captions(self, event: CaptionsStateEvent) -> None:
        self._active = event.is_active
        if not event.is_active:
            self._chunker.reset()

    async def accept_frame(self, frame: bytes) -> None:
        if not self._active:
            return
        pcm = self._chunker.push(frame)
        if not pcm:
            return
        duration_s = len(pcm) / 2 / self._sample_rate
        await self._bus.publish(
            AudioChunkEvent(
                chunk_id=str(uuid.uuid4()),
                pcm_s16le=pcm,
                sample_rate=self._sample_rate,
                started_at=time.monotonic() - duration_s,
                duration_s=duration_s,
            )
        )
