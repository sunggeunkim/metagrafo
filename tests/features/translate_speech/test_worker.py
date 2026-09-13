import asyncio
import threading

from core.event_bus import EventBus
from features.capture_audio.events import AudioChunkEvent
from features.operator_control.events import CaptionsStateEvent
from features.translate_speech.events import SubtitleEvent
from features.translate_speech.worker import TranslateWorker


def _chunk(chunk_id: str = "c1") -> AudioChunkEvent:
    return AudioChunkEvent(
        chunk_id=chunk_id,
        pcm_s16le=b"\x00\x10" * 1600,
        sample_rate=16000,
        started_at=1.0,
        duration_s=1.0,
    )


class FakeWhisper:
    def __init__(self, text: str = "Hello, everyone.") -> None:
        self.text = text
        self.calls: list[dict] = []

    def transcribe(self, _pcm: bytes, *, language: str, task: str, initial_prompt: str) -> str:
        self.calls.append(
            {"language": language, "task": task, "initial_prompt": initial_prompt}
        )
        return self.text


async def test_ko_to_en_uses_translate_task_and_publishes_english() -> None:
    bus = EventBus()
    whisper = FakeWhisper("God is good.")
    seen: list[SubtitleEvent] = []

    async def handler(event: SubtitleEvent) -> None:
        seen.append(event)

    bus.subscribe(SubtitleEvent, handler)
    worker = TranslateWorker(bus, whisper.transcribe, initial_prompt="Jesus, Amen.")
    task = asyncio.create_task(worker.run())
    await bus.publish(CaptionsStateEvent(is_active=True))
    await bus.publish(_chunk())
    await asyncio.sleep(0.1)
    task.cancel()
    assert [e.text for e in seen] == ["God is good."]
    assert whisper.calls == [
        {"language": "ko", "task": "translate", "initial_prompt": "Jesus, Amen."}
    ]
    assert seen[0].mode == "ko_to_en"
    assert seen[0].target_language == "en"


async def test_en_to_en_uses_transcribe_task() -> None:
    bus = EventBus()
    whisper = FakeWhisper("Hello, church.")
    seen: list[SubtitleEvent] = []

    async def handler(event: SubtitleEvent) -> None:
        seen.append(event)

    bus.subscribe(SubtitleEvent, handler)
    worker = TranslateWorker(bus, whisper.transcribe, mode="en_to_en")
    task = asyncio.create_task(worker.run())
    await bus.publish(CaptionsStateEvent(is_active=True))
    await bus.publish(_chunk())
    await asyncio.sleep(0.1)
    task.cancel()
    assert whisper.calls[0]["language"] == "en"
    assert whisper.calls[0]["task"] == "transcribe"
    assert seen[0].text == "Hello, church."


async def test_empty_whisper_output_is_not_published() -> None:
    bus = EventBus()
    whisper = FakeWhisper("   ")
    seen: list[SubtitleEvent] = []

    async def handler(event: SubtitleEvent) -> None:
        seen.append(event)

    bus.subscribe(SubtitleEvent, handler)
    worker = TranslateWorker(bus, whisper.transcribe)
    task = asyncio.create_task(worker.run())
    await bus.publish(CaptionsStateEvent(is_active=True))
    await bus.publish(_chunk())
    await asyncio.sleep(0.1)
    task.cancel()
    assert seen == []


async def test_inactive_discards_in_flight_result() -> None:
    bus = EventBus()
    started = threading.Event()
    release = threading.Event()

    def transcribe(_pcm: bytes, **_kwargs: object) -> str:
        started.set()
        release.wait(timeout=2)
        return "should not appear"

    seen: list[SubtitleEvent] = []

    async def handler(event: SubtitleEvent) -> None:
        seen.append(event)

    bus.subscribe(SubtitleEvent, handler)
    worker = TranslateWorker(bus, transcribe)
    task = asyncio.create_task(worker.run())
    await bus.publish(CaptionsStateEvent(is_active=True))
    await bus.publish(_chunk())
    assert started.wait(timeout=1)
    await bus.publish(CaptionsStateEvent(is_active=False))
    release.set()
    await asyncio.sleep(0.1)
    task.cancel()
    assert seen == []


async def test_full_queue_drops_oldest() -> None:
    bus = EventBus()
    started = threading.Event()
    release = threading.Event()
    texts: list[str] = []

    def transcribe(pcm: bytes, **_kwargs: object) -> str:
        started.set()
        release.wait(timeout=2)
        return pcm.decode("latin1")

    async def handler(event: SubtitleEvent) -> None:
        texts.append(event.text)

    bus.subscribe(SubtitleEvent, handler)
    worker = TranslateWorker(bus, transcribe, queue_size=4)
    task = asyncio.create_task(worker.run())
    await bus.publish(CaptionsStateEvent(is_active=True))
    await bus.publish(
        AudioChunkEvent(
            chunk_id="A",
            pcm_s16le=b"A",
            sample_rate=16000,
            started_at=1.0,
            duration_s=1.0,
        )
    )
    assert started.wait(timeout=1)
    for name in "BCDEF":
        await bus.publish(
            AudioChunkEvent(
                chunk_id=name,
                pcm_s16le=name.encode("ascii"),
                sample_rate=16000,
                started_at=1.0,
                duration_s=1.0,
            )
        )
    release.set()
    await asyncio.sleep(0.2)
    task.cancel()
    assert "B" not in texts
    assert "A" in texts
    assert "F" in texts
