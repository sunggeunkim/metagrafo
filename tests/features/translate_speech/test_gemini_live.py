from __future__ import annotations

import asyncio
from types import SimpleNamespace

from core.event_bus import EventBus
from features.capture_audio.events import ProgramAudioEvent
from features.operator_control.events import CaptionsStateEvent, ModeChangedEvent, TranslateMode
from features.translate_speech.events import SubtitleEvent
from features.translate_speech.gemini_live import GeminiLiveTranslator, echoes_target_language


class ScriptedSession:
    def __init__(self, messages: list[object]) -> None:
        self._messages = messages
        self.sent: list[object] = []

    async def __aenter__(self) -> ScriptedSession:
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def send_realtime_input(self, **_kwargs: object) -> None:
        return None

    async def receive(self):
        for message in self._messages:
            yield message


class HoldingSession(ScriptedSession):
    def __init__(self, messages: list[object] | None = None) -> None:
        super().__init__(messages or [])

    async def receive(self):
        for message in self._messages:
            yield message
        await asyncio.Event().wait()
        yield None


def _content(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        session_resumption_update=None,
        go_away=None,
        server_content=SimpleNamespace(
            output_transcription=SimpleNamespace(text=text),
            turn_complete=False,
        ),
    )


async def test_fragments_grow_then_a_sentence_commits() -> None:
    bus = EventBus()
    seen: list[SubtitleEvent] = []

    async def handler(event: SubtitleEvent) -> None:
        seen.append(event)

    bus.subscribe(SubtitleEvent, handler)
    calls = 0
    sessions = [
        ScriptedSession(
            [
                _content("Today's topical"),
                _content("sermon."),
                SimpleNamespace(
                    go_away=SimpleNamespace(time_left="10s"),
                    server_content=None,
                ),
            ]
        ),
        HoldingSession(),
    ]

    def connect(_mode: TranslateMode) -> ScriptedSession:
        nonlocal calls
        session = sessions[min(calls, len(sessions) - 1)]
        calls += 1
        return session

    translator = GeminiLiveTranslator(bus, connect=connect, api_key="secret")
    task = asyncio.create_task(translator.run())
    for _ in range(50):
        if calls >= 2:
            break
        await asyncio.sleep(0.02)
    task.cancel()
    with_cancel = asyncio.gather(task, return_exceptions=True)
    await with_cancel
    assert calls == 2
    assert [(event.text, event.provisional) for event in seen] == [
        ("Today's topical", True),
        ("Today's topical sermon.", False),
    ]


async def test_captions_off_drops_the_open_line() -> None:
    bus = EventBus()
    seen: list[SubtitleEvent] = []

    async def handler(event: SubtitleEvent) -> None:
        seen.append(event)

    bus.subscribe(SubtitleEvent, handler)
    translator = GeminiLiveTranslator(bus, connect=lambda _mode: HoldingSession())
    await translator.apply_transcript("Originally")
    await bus.publish(CaptionsStateEvent(is_active=False))
    await bus.publish(CaptionsStateEvent(is_active=True))
    await translator.apply_transcript("Today")
    assert [(event.text, event.provisional) for event in seen] == [
        ("Originally", True),
        ("Today", True),
    ]


async def test_session_end_commits_the_open_line_before_the_next_socket() -> None:
    bus = EventBus()
    seen: list[SubtitleEvent] = []

    async def handler(event: SubtitleEvent) -> None:
        seen.append(event)

    bus.subscribe(SubtitleEvent, handler)
    calls = 0
    sessions = [
        ScriptedSession([_content("Originally")]),
        HoldingSession([_content("Today")]),
    ]

    def connect(_mode: TranslateMode) -> ScriptedSession:
        nonlocal calls
        session = sessions[min(calls, len(sessions) - 1)]
        calls += 1
        return session

    translator = GeminiLiveTranslator(bus, connect=connect)
    task = asyncio.create_task(translator.run())
    for _ in range(50):
        if any(event.text == "Today" for event in seen):
            break
        await asyncio.sleep(0.02)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert [(event.text, event.provisional) for event in seen] == [
        ("Originally", True),
        ("Originally", False),
        ("Today", True),
    ]


def test_guest_mode_echoes_english() -> None:
    assert echoes_target_language(TranslateMode.KO_TO_EN) is False
    assert echoes_target_language(TranslateMode.EN_TO_EN) is True


async def test_guest_mode_reconnects_the_socket() -> None:
    bus = EventBus()
    modes: list[TranslateMode] = []

    def connect(mode: TranslateMode) -> HoldingSession:
        modes.append(mode)
        return HoldingSession()

    translator = GeminiLiveTranslator(bus, connect=connect)
    task = asyncio.create_task(translator.run())
    for _ in range(50):
        if modes:
            break
        await asyncio.sleep(0.02)
    await bus.publish(
        ModeChangedEvent(mode="en_to_en", source_language="en", target_language="en")
    )
    for _ in range(50):
        if len(modes) >= 2:
            break
        await asyncio.sleep(0.02)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert modes == [TranslateMode.KO_TO_EN, TranslateMode.EN_TO_EN]


async def test_full_audio_queue_keeps_the_newest_chunk() -> None:
    bus = EventBus()
    translator = GeminiLiveTranslator(bus, connect=lambda _mode: HoldingSession())
    for index in range(12):
        await bus.publish(ProgramAudioEvent(pcm_s16le=bytes([index]), sample_rate=16000))
    kept: list[bytes] = []
    while not translator._audio.empty():
        kept.append(translator._audio.get_nowait())
    assert kept == [bytes([index]) for index in range(2, 12)]
