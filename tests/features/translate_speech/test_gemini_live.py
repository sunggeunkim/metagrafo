from __future__ import annotations

import asyncio
from types import SimpleNamespace

from core.event_bus import EventBus
from features.capture_audio.events import ProgramAudioEvent
from features.operator_control.events import (
    CaptionsStateEvent,
    ModeChangedEvent,
    TranslateMode,
    WholeSentencesEvent,
    languages_for,
)
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


async def test_the_rest_of_a_cut_line_keeps_the_time_its_words_were_spoken() -> None:
    bus = EventBus()
    seen = _collect(bus)
    times = iter([1.0, 2.0, 3.0, 4.0, 5.0])
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True, clock=lambda: next(times)
    )
    await translator.apply_transcript(
        "Brothers and sisters, we give thanks for everyone who came from the north side of the city to worship with us, and we"
    )
    await translator.apply_transcript("pray for the sick.")
    assert [(e.text, e.spoken_at) for e in seen] == [
        ("Brothers and sisters, we give thanks for everyone who came from the north side of the city to worship with us,", 1.0),
        ("and we pray for the sick.", 1.0),
    ]


async def test_captions_off_while_a_cut_line_is_shown_drops_the_rest_of_it() -> None:
    bus = EventBus()
    seen = _collect(bus)

    async def stop_captions_once(event: SubtitleEvent) -> None:
        if len(seen) == 1:
            await bus.publish(CaptionsStateEvent(is_active=False))

    bus.subscribe(SubtitleEvent, stop_captions_once)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True
    )
    await translator.apply_transcript(
        "Brothers, we give thanks for the sisters who came from the north side of the city this morning"
    )
    await translator.apply_transcript("to worship with us and to share the meal after the service.")
    assert [(e.text, e.provisional) for e in seen] == [
        ("Brothers, we give thanks for the sisters who came from the north side of the city this morning to worship with us and to", False),
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


async def test_whole_sentences_keeps_words_that_are_doubled_in_good_grammar() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True
    )
    await translator.apply_transcript("He had had enough, and he knew that that night")
    await translator.apply_transcript("was the night God would call him.")
    assert [(e.text, e.provisional) for e in seen] == [
        ("He had had enough, and he knew that that night was the night God would call him.", False)
    ]


async def test_whole_sentences_keeps_phrases_that_carry_meaning() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True
    )
    await translator.apply_transcript("I mean, you know, God is faithful.")
    assert [(e.text, e.provisional) for e in seen] == [("I mean, you know, God is faithful.", False)]


async def test_whole_sentences_commits_the_open_line_after_a_quiet_spell() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True, quiet_s=0.05
    )
    await translator.apply_transcript("And so we pray")
    assert seen == []
    await asyncio.sleep(0.1)
    assert [(e.text, e.provisional) for e in seen] == [("And so we pray", False)]


async def test_whole_sentences_waits_again_after_each_fragment() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True, quiet_s=0.08
    )
    await translator.apply_transcript("And so")
    await asyncio.sleep(0.05)
    await translator.apply_transcript("we pray")
    await asyncio.sleep(0.05)
    assert seen == []
    await asyncio.sleep(0.1)
    assert [(e.text, e.provisional) for e in seen] == [("And so we pray", False)]


async def test_switching_whole_sentences_on_finishes_the_growing_line_as_it_started() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(bus, connect=lambda _mode: HoldingSession())
    await translator.apply_transcript("Today's topical")
    await bus.publish(WholeSentencesEvent(whole_sentences=True))
    await translator.apply_transcript("sermon.")
    await translator.apply_transcript("Yes.")
    assert [(e.text, e.provisional) for e in seen] == [
        ("Today's topical", True),
        ("Today's topical sermon.", False),
    ]


async def test_switching_whole_sentences_off_keeps_the_held_sentence() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True
    )
    await translator.apply_transcript("Yes.")
    await bus.publish(WholeSentencesEvent(whole_sentences=False))
    await translator.apply_transcript("Today we read Romans.")
    await translator.apply_transcript("And")
    assert [(e.text, e.provisional) for e in seen] == [
        ("Yes. Today we read Romans.", False),
        ("And", True),
    ]


async def test_a_finished_sentence_stops_its_quiet_wait_so_the_next_line_is_not_cut_short() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True, quiet_s=0.05
    )
    await translator.apply_transcript("Today we read")
    await translator.apply_transcript("from the letter to the Romans.")
    await bus.publish(WholeSentencesEvent(whole_sentences=False))
    await translator.apply_transcript("And so")
    await asyncio.sleep(0.1)
    assert [(e.text, e.provisional) for e in seen] == [
        ("Today we read from the letter to the Romans.", False),
        ("And so", True),
    ]


async def test_an_empty_turn_starts_no_quiet_wait_that_could_cut_the_next_line_short() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=True, quiet_s=0.05
    )
    await translator.apply_transcript("", turn_complete=True)
    await bus.publish(WholeSentencesEvent(whole_sentences=False))
    await translator.apply_transcript("And so")
    await asyncio.sleep(0.1)
    assert [(e.text, e.provisional) for e in seen] == [("And so", True)]


async def test_whole_sentences_off_shows_the_line_as_spoken() -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus, connect=lambda _mode: HoldingSession(), whole_sentences=False, quiet_s=0.05
    )
    await translator.apply_transcript("Uh, yes,")
    await translator.apply_transcript("the the Lord.")
    await translator.apply_transcript("And so")
    await asyncio.sleep(0.1)
    assert [(e.text, e.provisional) for e in seen] == [
        ("Uh, yes,", True),
        ("Uh, yes, the the Lord.", False),
        ("And so", True),
    ]


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def test_clean_up_wording_shows_the_cleaned_line() -> None:
    bus = EventBus()
    seen = _collect(bus)

    async def cleanup(line: str, context: list[str]) -> str:
        return "Paul says nothing can separate us from the love of God."

    translator = GeminiLiveTranslator(
        bus,
        connect=lambda _mode: HoldingSession(),
        whole_sentences=True,
        clean_up_wording=True,
        cleanup=cleanup,
    )
    await translator.apply_transcript(
        "And so Paul, what he says is, nothing can separate us from the love of God."
    )
    await _settle()
    assert [(e.text, e.provisional) for e in seen] == [
        ("Paul says nothing can separate us from the love of God.", False)
    ]


async def test_clean_up_wording_gives_the_model_the_last_three_shown_lines() -> None:
    bus = EventBus()
    seen = _collect(bus)
    asked: list[tuple[str, list[str]]] = []

    async def cleanup(line: str, context: list[str]) -> str:
        asked.append((line, context))
        return line.upper()

    translator = GeminiLiveTranslator(
        bus,
        connect=lambda _mode: HoldingSession(),
        whole_sentences=True,
        clean_up_wording=True,
        cleanup=cleanup,
    )
    for sentence in (
        "Today we read from the letter to the Romans.",
        "Paul writes to the church in the city of Rome.",
        "He has never visited them in person before.",
        "So he tells them who he is and what he believes.",
        "He begins with the good news about Jesus.",
    ):
        await translator.apply_transcript(sentence)
        await _settle()
    assert asked[0] == ("Today we read from the letter to the Romans.", [])
    assert asked[-1] == (
        "He begins with the good news about Jesus.",
        [
            "PAUL WRITES TO THE CHURCH IN THE CITY OF ROME.",
            "HE HAS NEVER VISITED THEM IN PERSON BEFORE.",
            "SO HE TELLS THEM WHO HE IS AND WHAT HE BELIEVES.",
        ],
    )


async def test_a_slow_cleanup_shows_the_line_as_heard_and_logs_why(caplog) -> None:
    bus = EventBus()
    seen = _collect(bus)

    async def cleanup(line: str, context: list[str]) -> str:
        await asyncio.sleep(5)
        return "Too late."

    translator = GeminiLiveTranslator(
        bus,
        connect=lambda _mode: HoldingSession(),
        whole_sentences=True,
        clean_up_wording=True,
        cleanup=cleanup,
        cleanup_timeout_s=0.05,
    )
    with caplog.at_level("WARNING"):
        await translator.apply_transcript("And so we come before the Lord in prayer.")
        await asyncio.sleep(0.2)
    assert [(e.text, e.provisional) for e in seen] == [
        ("And so we come before the Lord in prayer.", False)
    ]
    assert "cleanup skipped: no reply within 0.05 s" in caplog.text


async def test_a_failed_cleanup_shows_the_line_as_heard_and_logs_why_without_the_key(caplog) -> None:
    bus = EventBus()
    seen = _collect(bus)

    async def cleanup(line: str, context: list[str]) -> str:
        raise RuntimeError("429 quota exhausted for key secret-key-7")

    translator = GeminiLiveTranslator(
        bus,
        connect=lambda _mode: HoldingSession(),
        api_key="secret-key-7",
        whole_sentences=True,
        clean_up_wording=True,
        cleanup=cleanup,
    )
    with caplog.at_level("WARNING"):
        await translator.apply_transcript("And so we come before the Lord in prayer.")
        await _settle()
    assert [(e.text, e.provisional) for e in seen] == [
        ("And so we come before the Lord in prayer.", False)
    ]
    assert "cleanup skipped: 429 quota exhausted for key [key]" in caplog.text
    assert "secret-key-7" not in caplog.text


async def test_cleaned_lines_show_in_the_order_they_were_spoken() -> None:
    bus = EventBus()
    seen = _collect(bus)
    first_may_finish = asyncio.Event()

    async def cleanup(line: str, context: list[str]) -> str:
        if line.startswith("Today"):
            await first_may_finish.wait()
        return line.upper()

    translator = GeminiLiveTranslator(
        bus,
        connect=lambda _mode: HoldingSession(),
        whole_sentences=True,
        clean_up_wording=True,
        cleanup=cleanup,
    )
    # The socket keeps reading while a line is being cleaned.
    await asyncio.wait_for(
        translator.apply_transcript("Today we read from the letter to the Romans."), timeout=0.5
    )
    await asyncio.wait_for(
        translator.apply_transcript("Paul writes to the church in the city of Rome."), timeout=0.5
    )
    await _settle()
    assert seen == []
    first_may_finish.set()
    await _settle()
    assert [e.text for e in seen] == [
        "TODAY WE READ FROM THE LETTER TO THE ROMANS.",
        "PAUL WRITES TO THE CHURCH IN THE CITY OF ROME.",
    ]


async def test_captions_off_drops_a_line_still_being_cleaned_and_stops_its_call() -> None:
    bus = EventBus()
    seen = _collect(bus)
    first_may_finish = asyncio.Event()
    stopped: list[str] = []

    async def cleanup(line: str, context: list[str]) -> str:
        if line.startswith("Today"):
            try:
                await first_may_finish.wait()
            except asyncio.CancelledError:
                stopped.append(line)
                raise
        return line.upper()

    translator = GeminiLiveTranslator(
        bus,
        connect=lambda _mode: HoldingSession(),
        whole_sentences=True,
        clean_up_wording=True,
        cleanup=cleanup,
    )
    await translator.apply_transcript("Today we read from the letter to the Romans.")
    await _settle()
    await bus.publish(CaptionsStateEvent(is_active=False))
    await bus.publish(CaptionsStateEvent(is_active=True))
    await translator.apply_transcript("Paul writes to the church in the city of Rome.")
    first_may_finish.set()
    await _settle()
    assert [e.text for e in seen] == ["PAUL WRITES TO THE CHURCH IN THE CITY OF ROME."]
    assert stopped == ["Today we read from the letter to the Romans."]


async def test_switching_clean_up_wording_off_shows_the_next_line_as_heard_after_the_one_being_cleaned() -> None:
    from features.operator_control.events import CleanUpWordingEvent

    bus = EventBus()
    seen = _collect(bus)
    first_may_finish = asyncio.Event()

    async def cleanup(line: str, context: list[str]) -> str:
        await first_may_finish.wait()
        return line.upper()

    translator = GeminiLiveTranslator(
        bus,
        connect=lambda _mode: HoldingSession(),
        whole_sentences=True,
        clean_up_wording=True,
        cleanup=cleanup,
    )
    await translator.apply_transcript("Today we read")
    await bus.publish(CleanUpWordingEvent(clean_up_wording=False))
    await translator.apply_transcript("from the letter to the Romans.")
    await translator.apply_transcript("Paul writes to the church in the city of Rome.")
    await _settle()
    assert seen == []
    first_may_finish.set()
    await _settle()
    assert [e.text for e in seen] == [
        "TODAY WE READ FROM THE LETTER TO THE ROMANS.",
        "Paul writes to the church in the city of Rome.",
    ]


async def test_clean_up_wording_leaves_lines_alone_while_whole_sentences_is_off() -> None:
    bus = EventBus()
    seen = _collect(bus)
    asked: list[str] = []

    async def cleanup(line: str, context: list[str]) -> str:
        asked.append(line)
        return line.upper()

    translator = GeminiLiveTranslator(
        bus,
        connect=lambda _mode: HoldingSession(),
        whole_sentences=False,
        clean_up_wording=True,
        cleanup=cleanup,
    )
    await translator.apply_transcript("Uh, today we read")
    await translator.apply_transcript("from Romans.")
    await _settle()
    assert [(e.text, e.provisional) for e in seen] == [
        ("Uh, today we read", True),
        ("Uh, today we read from Romans.", False),
    ]
    assert asked == []


async def test_clean_up_wording_without_a_key_shows_the_line_as_heard_and_logs_why(caplog) -> None:
    bus = EventBus()
    seen = _collect(bus)
    translator = GeminiLiveTranslator(
        bus,
        connect=lambda _mode: HoldingSession(),
        whole_sentences=True,
        clean_up_wording=True,
        cleanup=None,
    )
    with caplog.at_level("WARNING"):
        await translator.apply_transcript("And so we come before the Lord in prayer.")
        await _settle()
    assert [e.text for e in seen] == ["And so we come before the Lord in prayer."]
    assert "cleanup skipped: no Gemini key" in caplog.text


async def test_a_line_cleaned_across_a_mode_change_keeps_the_mode_it_was_spoken_in() -> None:
    bus = EventBus()
    seen = _collect(bus)
    first_may_finish = asyncio.Event()

    async def cleanup(line: str, context: list[str]) -> str:
        await first_may_finish.wait()
        return line.upper()

    translator = GeminiLiveTranslator(
        bus,
        connect=lambda _mode: HoldingSession(),
        whole_sentences=True,
        clean_up_wording=True,
        cleanup=cleanup,
    )
    await translator.apply_transcript("Today we read from the letter to the Romans.")
    await bus.publish(ModeChangedEvent(mode="en_to_en", source_language="en", target_language="en"))
    first_may_finish.set()
    await _settle()
    assert [(e.text, e.mode, e.source_language) for e in seen] == [
        ("TODAY WE READ FROM THE LETTER TO THE ROMANS.", "ko_to_en", "ko")
    ]


async def test_an_empty_cleanup_reply_shows_the_line_as_heard(caplog) -> None:
    bus = EventBus()
    seen = _collect(bus)

    async def cleanup(line: str, context: list[str]) -> str:
        return "  "

    translator = GeminiLiveTranslator(
        bus,
        connect=lambda _mode: HoldingSession(),
        whole_sentences=True,
        clean_up_wording=True,
        cleanup=cleanup,
    )
    with caplog.at_level("WARNING"):
        await translator.apply_transcript("And so we come before the Lord in prayer.")
        await _settle()
    assert [e.text for e in seen] == ["And so we come before the Lord in prayer."]
    assert "cleanup skipped: empty reply" in caplog.text


async def test_a_growing_line_waits_for_the_line_still_being_cleaned_before_it() -> None:
    bus = EventBus()
    seen = _collect(bus)
    first_may_finish = asyncio.Event()

    async def cleanup(line: str, context: list[str]) -> str:
        await first_may_finish.wait()
        return line.upper()

    translator = GeminiLiveTranslator(
        bus,
        connect=lambda _mode: HoldingSession(),
        whole_sentences=True,
        clean_up_wording=True,
        cleanup=cleanup,
    )
    await translator.apply_transcript("Today we read from the letter to the Romans.")
    await bus.publish(WholeSentencesEvent(whole_sentences=False))
    await translator.apply_transcript("And so")
    await _settle()
    assert seen == []
    first_may_finish.set()
    await _settle()
    await translator.apply_transcript("we pray")
    assert [(e.text, e.provisional) for e in seen] == [
        ("TODAY WE READ FROM THE LETTER TO THE ROMANS.", False),
        ("And so we pray", True),
    ]
