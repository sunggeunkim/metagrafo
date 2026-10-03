import numpy as np

from core.event_bus import EventBus
from features.capture_audio.chunker import SpeechChunker
from features.capture_audio.events import AudioChunkEvent, ProgramAudioEvent
from features.capture_audio.gate import CaptureGate
from features.operator_control.events import CaptionsStateEvent, VadPausesEvent

FRAME = 512
SPEECH = np.full(FRAME, 8000, dtype=np.int16).tobytes()
SILENCE = np.zeros(FRAME, dtype=np.int16).tobytes()


def _energy(frame: bytes) -> bool:
    return float(np.abs(np.frombuffer(frame, dtype=np.int16)).mean()) > 1000


def _gate(bus: EventBus, *, stream_program_audio: bool = False) -> CaptureGate:
    chunker = SpeechChunker(
        sample_rate=16000,
        frame_samples=FRAME,
        preroll_ms=200,
        min_silence_ms=800,
        max_utterance_s=12,
        min_speech_ms=250,
        is_speech=_energy,
    )
    return CaptureGate(
        bus=bus,
        chunker=chunker,
        sample_rate=16000,
        stream_program_audio=stream_program_audio,
    )


async def test_gemini_live_publishes_program_audio_before_a_vad_cut() -> None:
    bus = EventBus()
    seen: list[ProgramAudioEvent] = []

    async def handler(event: ProgramAudioEvent) -> None:
        seen.append(event)

    bus.subscribe(ProgramAudioEvent, handler)
    gate = _gate(bus, stream_program_audio=True)
    await bus.publish(CaptionsStateEvent(is_active=True))
    for _ in range(4):
        await gate.accept_frame(SILENCE)
    assert len(seen) == 1
    assert seen[0].sample_rate == 16000
    assert len(seen[0].pcm_s16le) == 3200


async def test_whisper_capture_does_not_publish_program_audio() -> None:
    bus = EventBus()
    seen: list[ProgramAudioEvent] = []

    async def handler(event: ProgramAudioEvent) -> None:
        seen.append(event)

    bus.subscribe(ProgramAudioEvent, handler)
    gate = _gate(bus)
    await bus.publish(CaptionsStateEvent(is_active=True))
    for _ in range(8):
        await gate.accept_frame(SILENCE)
    assert seen == []


async def test_inactive_capture_drops_frames_and_publishes_nothing() -> None:
    bus = EventBus()
    seen: list[AudioChunkEvent] = []

    async def handler(event: AudioChunkEvent) -> None:
        seen.append(event)

    bus.subscribe(AudioChunkEvent, handler)
    gate = _gate(bus)
    await bus.publish(CaptionsStateEvent(is_active=False))
    for _ in range(16):
        await gate.accept_frame(SPEECH)
    for _ in range(30):
        await gate.accept_frame(SILENCE)
    assert seen == []


async def test_active_capture_publishes_audio_chunk_for_speech() -> None:
    bus = EventBus()
    seen: list[AudioChunkEvent] = []

    async def handler(event: AudioChunkEvent) -> None:
        seen.append(event)

    bus.subscribe(AudioChunkEvent, handler)
    gate = _gate(bus)
    await bus.publish(CaptionsStateEvent(is_active=True))
    for _ in range(16):
        await gate.accept_frame(SPEECH)
    for _ in range(30):
        await gate.accept_frame(SILENCE)
    assert len(seen) == 1
    assert seen[0].sample_rate == 16000
    assert seen[0].duration_s > 0.25
    assert len(seen[0].pcm_s16le) > 0


async def test_two_pauses_from_the_booth_hold_the_first_phrase() -> None:
    bus = EventBus()
    seen: list[AudioChunkEvent] = []

    async def handler(event: AudioChunkEvent) -> None:
        seen.append(event)

    bus.subscribe(AudioChunkEvent, handler)
    gate = _gate(bus)
    await bus.publish(VadPausesEvent(vad_pauses=2))
    for _ in range(16):
        await gate.accept_frame(SPEECH)
    for _ in range(30):
        await gate.accept_frame(SILENCE)
    assert seen == []
    for _ in range(16):
        await gate.accept_frame(SPEECH)
    for _ in range(30):
        await gate.accept_frame(SILENCE)
    assert len(seen) == 2
