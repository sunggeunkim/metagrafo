from core.event_bus import EventBus
from features.capture_audio.events import AudioChunkEvent
from features.translate_speech.events import SubtitleEvent
from features.translate_speech.placeholder import register_placeholder


async def test_audio_chunk_becomes_placeholder_subtitle() -> None:
    bus = EventBus()
    seen: list[SubtitleEvent] = []

    async def handler(event: SubtitleEvent) -> None:
        seen.append(event)

    bus.subscribe(SubtitleEvent, handler)
    register_placeholder(bus, mode="ko_to_en")
    await bus.publish(
        AudioChunkEvent(
            chunk_id="c1",
            pcm_s16le=b"\x00\x10" * 16000,
            sample_rate=16000,
            started_at=1.0,
            duration_s=2.5,
        )
    )
    assert len(seen) == 1
    assert seen[0].text == "speech 2.5s"
    assert seen[0].chunk_id == "c1"
    assert seen[0].mode == "ko_to_en"
