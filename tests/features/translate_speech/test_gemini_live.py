from __future__ import annotations

import asyncio
from types import SimpleNamespace

from core.event_bus import EventBus
from features.capture_audio.events import ProgramAudioEvent
from features.operator_control.events import CaptionsStateEvent, ModeChangedEvent, TranslateMode, languages_for
from features.translate_speech.events import SubtitleEvent
from features.translate_speech.gemini_live import GeminiLiveTranslator, echoes_target_language, gemini_connect


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
    assert echoes_target_language(TranslateMode.EN_TO_KO) is False


def test_korean_guest_mode_targets_korean() -> None:
    assert languages_for(TranslateMode.EN_TO_KO) == ("en", "ko")


async def test_korean_guest_mode_publishes_korean_subtitles() -> None:
    bus = EventBus()
    seen: list[SubtitleEvent] = []

    async def handler(event: SubtitleEvent) -> None:
        seen.append(event)

    bus.subscribe(SubtitleEvent, handler)
    translator = GeminiLiveTranslator(bus, connect=lambda _mode: HoldingSession(), mode="en_to_ko")
    await translator.apply_transcript("안녕하세요, 여러분.")
    assert [(e.text, e.source_language, e.target_language, e.mode) for e in seen] == [
        ("안녕하세요, 여러분.", "en", "ko", "en_to_ko")
    ]


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


async def test_finished_line_is_stamped_when_its_first_fragment_arrived() -> None:
    bus = EventBus()
    seen: list[SubtitleEvent] = []

    async def handler(event: SubtitleEvent) -> None:
        seen.append(event)

    bus.subscribe(SubtitleEvent, handler)
    # Each fragment reads the clock, then each publish reads it again.
    ticks = iter([100.0, 100.1, 104.0, 104.1])
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), clock=lambda: next(ticks)
    )
    await translator.apply_transcript("Today's topical")
    await translator.apply_transcript("sermon.")
    assert [(e.provisional, e.spoken_at) for e in seen] == [(True, 100.0), (False, 100.0)]
    assert seen[-1].created_at == 104.1


def test_live_socket_uses_the_api_version_that_serves_the_translate_model(monkeypatch) -> None:
    # google-genai 3.0's default API version answers the Live translate preview model with HTTP 404.
    from google import genai

    built: dict[str, object] = {}

    class RecordingClient:
        def __init__(self, **kwargs: object) -> None:
            built.update(kwargs)
            self.aio = SimpleNamespace(live=SimpleNamespace(connect=lambda **kw: kw))

    monkeypatch.setattr(genai, "Client", RecordingClient)
    gemini_connect("test-key")(TranslateMode.EN_TO_KO)
    options = built.get("http_options")
    api_version = options.get("api_version") if isinstance(options, dict) else getattr(options, "api_version", None)
    assert api_version == "v1beta"


def _collect(bus: EventBus) -> list[SubtitleEvent]:
    seen: list[SubtitleEvent] = []

    async def handler(event: SubtitleEvent) -> None:
        seen.append(event)

    bus.subscribe(SubtitleEvent, handler)
    return seen


async def test_whole_sentences_shows_only_the_finished_line() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True
    )
    await translator.apply_transcript("Today's topical")
    await translator.apply_transcript("sermon is about prayer.")
    assert [(e.text, e.provisional) for e in seen] == [
        ("Today's topical sermon is about prayer.", False)
    ]


async def test_whole_sentences_joins_a_short_sentence_to_the_next() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True
    )
    await translator.apply_transcript("Yes.")
    await translator.apply_transcript("Today we read from Romans eight.")
    assert [(e.text, e.provisional) for e in seen] == [
        ("Yes. Today we read from Romans eight.", False)
    ]


async def test_whole_sentences_commits_a_long_line_when_the_turn_ends() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True
    )
    await translator.apply_transcript("And so we come before the Lord")
    await translator.apply_transcript("", turn_complete=True)
    assert [(e.text, e.provisional) for e in seen] == [
        ("And so we come before the Lord", False)
    ]


async def test_whole_sentences_cuts_a_long_line_at_the_last_comma() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True
    )
    await translator.apply_transcript(
        "When we gather on Sunday morning to pray for our neighbours and our city,"
    )
    await translator.apply_transcript(
        "we remember that the Lord hears every word we say and answers in his time"
    )
    await translator.apply_transcript("and in his way.")
    assert [(e.text, e.provisional) for e in seen] == [
        ("When we gather on Sunday morning to pray for our neighbours and our city,", False),
        ("we remember that the Lord hears every word we say and answers in his time and in his way.", False),
    ]


async def test_whole_sentences_cuts_a_long_line_without_a_comma_between_words() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True
    )
    await translator.apply_transcript(
        "We give thanks for the brothers and sisters who came from the north side of the city this morning"
    )
    await translator.apply_transcript("to worship with us and to share the meal")
    await translator.apply_transcript("after the service.")
    assert [(e.text, e.provisional) for e in seen] == [
        ("We give thanks for the brothers and sisters who came from the north side of the city this morning to worship with us and", False),
        ("to share the meal after the service.", False),
    ]


async def test_whole_sentences_does_not_cut_at_a_comma_that_leaves_a_stub() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True
    )
    await translator.apply_transcript(
        "Brothers, we give thanks for the sisters who came from the north side of the city this morning"
    )
    await translator.apply_transcript("to worship with us and to share the meal after the service.")
    assert [(e.text, e.provisional) for e in seen] == [
        ("Brothers, we give thanks for the sisters who came from the north side of the city this morning to worship with us and to", False),
        ("share the meal after the service.", False),
    ]


async def test_whole_sentences_removes_fillers() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True
    )
    await translator.apply_transcript("Uh, today we begin")
    await translator.apply_transcript("with um prayer, hmm.")
    assert [(e.text, e.provisional) for e in seen] == [("Today we begin with prayer.", False)]


async def test_whole_sentences_removes_immediate_stutters() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True
    )
    await translator.apply_transcript("We— we read the")
    await translator.apply_transcript("the letter to the church.")
    assert [(e.text, e.provisional) for e in seen] == [("We read the letter to the church.", False)]


async def test_whole_sentences_keeps_phrases_that_carry_meaning() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True
    )
    await translator.apply_transcript("I mean, you know, God is faithful.")
    assert [(e.text, e.provisional) for e in seen] == [("I mean, you know, God is faithful.", False)]
