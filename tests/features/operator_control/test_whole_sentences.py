import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from core.event_bus import EventBus
from core.settings import Settings
from features.operator_control.events import WholeSentencesEvent
from features.translate_speech.events import SubtitleEvent
from main import create_app


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("WHOLE_SENTENCES", raising=False)
    path = tmp_path / ".env"
    path.write_text("AUDIO_DEVICE_NAME=Booth Mic\n", encoding="utf-8")
    return path


class _SpeakingSession:
    """A Gemini socket that says one sentence in two fragments, then stays open."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def send_realtime_input(self, **_kwargs: object) -> None:
        return None

    async def receive(self):
        for text in ("Today's topical", "sermon is about prayer."):
            yield SimpleNamespace(
                go_away=None,
                server_content=SimpleNamespace(
                    output_transcription=SimpleNamespace(text=text), turn_complete=False
                ),
            )
        await asyncio.Event().wait()


def _captions(bus: EventBus) -> list[tuple[str, bool]]:
    seen: list[tuple[str, bool]] = []

    async def handler(event: SubtitleEvent) -> None:
        seen.append((event.text, event.provisional))

    bus.subscribe(SubtitleEvent, handler)
    return seen


def _wait_for_a_finished_line(client: TestClient, seen: list[tuple[str, bool]]) -> None:
    for _ in range(100):
        if any(not provisional for _text, provisional in seen):
            return
        client.get("/health")
        time.sleep(0.01)


def _speaking(_api_key: str):
    return lambda _mode: _SpeakingSession()


def _as_heard(_api_key: str, _model: str):
    """Clean up wording is on by default; this stands in for the text model, offline."""

    async def clean(line: str, _shown: list[str], _said: str) -> str:
        return line

    return clean


def _app(env_file, **kwargs):
    settings = Settings(caption_engine="whisper", gemini_api_key="", _env_file=env_file)
    return create_app(settings=settings, env_file=env_file, gemini_cleanup_factory=_as_heard, **kwargs)


def test_whole_sentences_starts_on(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        assert client.get("/health").json()["whole_sentences"] is True
        assert client.get("/mode").json()["whole_sentences"] is True


def test_booth_page_has_a_whole_sentences_checkbox(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        html = client.get("/control").text
    assert '<input id="whole-sentences" type="checkbox"' in html
    assert "/whole-sentences" in html


def test_switching_off_tells_gemini_and_keeps_it_for_next_time(env_file) -> None:
    bus = EventBus()
    seen: list[bool] = []

    async def handler(event: WholeSentencesEvent) -> None:
        seen.append(event.whole_sentences)

    bus.subscribe(WholeSentencesEvent, handler)
    with TestClient(_app(env_file, bus=bus)) as client:
        response = client.put("/whole-sentences", json={"whole_sentences": False})
        assert response.status_code == 200
        assert response.json()["whole_sentences"] is False
        assert client.get("/health").json()["whole_sentences"] is False
    assert seen == [False]
    assert env_file.read_text(encoding="utf-8") == "AUDIO_DEVICE_NAME=Booth Mic\nWHOLE_SENTENCES=false\n"
    with TestClient(_app(env_file)) as client:
        assert client.get("/health").json()["whole_sentences"] is False


def test_gemini_starts_with_whole_sentences_on(env_file) -> None:
    bus = EventBus()
    seen = _captions(bus)
    settings = Settings(caption_engine="gemini_live", gemini_api_key="saved-key", _env_file=env_file)
    app = create_app(
        bus=bus,
        settings=settings,
        env_file=env_file,
        gemini_connect_factory=_speaking,
        gemini_cleanup_factory=_as_heard,
    )
    with TestClient(app) as client:
        _wait_for_a_finished_line(client, seen)
    assert seen == [("Today's topical sermon is about prayer.", False)]


def test_a_pasted_key_starts_gemini_with_whole_sentences_on(env_file, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("CAPTION_ENGINE", raising=False)
    bus = EventBus()
    seen = _captions(bus)
    with TestClient(_app(env_file, bus=bus, gemini_connect_factory=_speaking)) as client:
        client.put("/gemini-key", json={"api_key": "pasted-key-42"})
        _wait_for_a_finished_line(client, seen)
    assert seen == [("Today's topical sermon is about prayer.", False)]


def test_gemini_starts_with_the_choice_saved_on_the_booth(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        client.put("/whole-sentences", json={"whole_sentences": False})
    bus = EventBus()
    seen = _captions(bus)
    settings = Settings(caption_engine="gemini_live", gemini_api_key="saved-key", _env_file=env_file)
    app = create_app(
        bus=bus,
        settings=settings,
        env_file=env_file,
        gemini_connect_factory=_speaking,
        gemini_cleanup_factory=_as_heard,
    )
    with TestClient(app) as client:
        _wait_for_a_finished_line(client, seen)
    assert seen == [
        ("Today's topical", True),
        ("Today's topical sermon is about prayer.", False),
    ]
