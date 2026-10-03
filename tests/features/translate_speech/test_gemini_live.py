from __future__ import annotations

import asyncio
from types import SimpleNamespace

from core.event_bus import EventBus
from features.operator_control.events import CaptionsStateEvent
from features.translate_speech.events import SubtitleEvent
from features.translate_speech.gemini_live import GeminiLiveTranslator


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
    def __init__(self) -> None:
        super().__init__([])

    async def receive(self):
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

    def connect() -> ScriptedSession:
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


async def test_captions_off_commits_the_open_line() -> None:
    bus = EventBus()
    seen: list[SubtitleEvent] = []

    async def handler(event: SubtitleEvent) -> None:
        seen.append(event)

    bus.subscribe(SubtitleEvent, handler)
    translator = GeminiLiveTranslator(bus, connect=HoldingSession)
    await translator.apply_transcript("Originally")
    await bus.publish(CaptionsStateEvent(is_active=False))
    assert [(event.text, event.provisional) for event in seen] == [
        ("Originally", True),
        ("Originally", False),
    ]
