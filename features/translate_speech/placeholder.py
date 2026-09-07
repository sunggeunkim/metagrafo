from __future__ import annotations

import time

from core.event_bus import EventBus
from features.capture_audio.events import AudioChunkEvent
from features.operator_control.events import ModeChangedEvent, TranslateMode, languages_for
from features.translate_speech.events import SubtitleEvent


def register_placeholder(bus: EventBus, *, mode: str = "ko_to_en") -> None:
    current = {"mode": mode}

    async def on_mode(event: ModeChangedEvent) -> None:
        current["mode"] = event.mode

    async def on_chunk(event: AudioChunkEvent) -> None:
        try:
            translate_mode = TranslateMode(current["mode"])
        except ValueError:
            translate_mode = TranslateMode.KO_TO_EN
        source, target = languages_for(translate_mode)
        await bus.publish(
            SubtitleEvent(
                chunk_id=event.chunk_id,
                text=f"speech {event.duration_s:.1f}s",
                source_language=source,
                target_language=target,
                mode=translate_mode.value,
                created_at=time.time(),
                duration_s=event.duration_s,
            )
        )

    bus.subscribe(ModeChangedEvent, on_mode)
    bus.subscribe(AudioChunkEvent, on_chunk)
