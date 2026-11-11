import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from core.event_bus import EventBus
from core.settings import GLOSSARY_DEFAULT, Settings
from features.operator_control.events import GlossaryEvent
from main import create_app

MINE = "Nirvana → 열반\nkarma → 업보"


@pytest.fixture
def env_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text("AUDIO_DEVICE_NAME=Booth Mic\n", encoding="utf-8")
    return path


def _app(env_file, **kwargs):
    settings = Settings(caption_engine="whisper", gemini_api_key="", _env_file=env_file)
    return create_app(settings=settings, env_file=env_file, **kwargs)


def _glossary_file(env_file):
    return env_file.parent / "glossary.txt"


def test_glossary_starts_as_the_church_default(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        body = client.get("/mode").json()
    assert body["glossary"] == GLOSSARY_DEFAULT
    assert body["glossary_default"] == GLOSSARY_DEFAULT


def test_a_saved_glossary_is_announced_and_kept_beside_env_for_next_time(env_file) -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler(event: GlossaryEvent) -> None:
        seen.append(event.text)

    bus.subscribe(GlossaryEvent, handler)
    with TestClient(_app(env_file, bus=bus)) as client:
        response = client.put("/glossary", json={"text": f"  {MINE}\n"})
        assert response.status_code == 200
        assert response.json()["glossary"] == MINE
    assert seen == [MINE]
    assert _glossary_file(env_file).read_text(encoding="utf-8") == MINE
    assert env_file.read_text(encoding="utf-8") == "AUDIO_DEVICE_NAME=Booth Mic\n"
    with TestClient(_app(env_file)) as client:
        assert client.get("/mode").json()["glossary"] == MINE


def test_saving_an_empty_glossary_turns_it_off_and_it_stays_off(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        assert client.put("/glossary", json={"text": "  "}).json()["glossary"] == ""
    with TestClient(_app(env_file)) as client:
        assert client.get("/mode").json()["glossary"] == ""


def test_reset_goes_back_to_the_default_and_stays_there(env_file) -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler(event: GlossaryEvent) -> None:
        seen.append(event.text)

    bus.subscribe(GlossaryEvent, handler)
    with TestClient(_app(env_file, bus=bus)) as client:
        client.put("/glossary", json={"text": MINE})
        response = client.delete("/glossary")
        assert response.status_code == 200
        assert response.json()["glossary"] == GLOSSARY_DEFAULT
    assert seen == [MINE, GLOSSARY_DEFAULT]
    assert not _glossary_file(env_file).exists()
    with TestClient(_app(env_file)) as client:
        assert client.get("/mode").json()["glossary"] == GLOSSARY_DEFAULT


def test_a_glossary_too_long_for_every_cleanup_call_is_refused(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        assert client.put("/glossary", json={"text": "x" * 3001}).status_code == 422
        assert client.get("/mode").json()["glossary"] == GLOSSARY_DEFAULT


def test_the_file_is_the_only_source_so_an_env_value_is_ignored(env_file, monkeypatch) -> None:
    monkeypatch.setenv("GLOSSARY", "From the shell, never saved on booth control.")
    with TestClient(_app(env_file)) as client:
        assert client.get("/mode").json()["glossary"] == GLOSSARY_DEFAULT


def test_a_hand_edited_file_over_the_limit_is_cut_to_it(env_file, caplog) -> None:
    _glossary_file(env_file).write_text("x" * 5000, encoding="utf-8")
    with TestClient(_app(env_file)) as client:
        assert client.get("/mode").json()["glossary"] == "x" * 3000
    assert "glossary.txt" in caplog.text


def test_an_unreadable_file_falls_back_to_the_default(env_file, caplog) -> None:
    _glossary_file(env_file).write_bytes(b"\xff\xfe\x00 not utf-8 \x81")
    with TestClient(_app(env_file)) as client:
        assert client.get("/mode").json()["glossary"] == GLOSSARY_DEFAULT
    assert "glossary.txt" in caplog.text


class _SpeakingSession:
    """A Gemini socket that says one sentence, then stays open."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def send_realtime_input(self, **_kwargs: object) -> None:
        return None

    async def receive(self):
        yield SimpleNamespace(
            go_away=None,
            server_content=SimpleNamespace(
                output_transcription=SimpleNamespace(text="Today we talk about suffering."),
                turn_complete=False,
            ),
        )
        await asyncio.Event().wait()


def test_the_saved_glossary_reaches_the_cleanup_model(env_file, monkeypatch) -> None:
    from google import genai

    instructions: list[str] = []

    class RecordingClient:
        def __init__(self, **_kwargs: object) -> None:
            async def generate_content(**kw: object) -> SimpleNamespace:
                instructions.append(kw["config"].system_instruction)
                return SimpleNamespace(text="Today we talk about suffering.")

            self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    monkeypatch.setattr(genai, "Client", RecordingClient)
    _glossary_file(env_file).write_text(MINE, encoding="utf-8")
    settings = Settings(caption_engine="gemini_live", gemini_api_key="saved-key", _env_file=env_file)
    app = create_app(
        settings=settings,
        env_file=env_file,
        gemini_connect_factory=lambda _key: (lambda _mode: _SpeakingSession()),
    )
    with TestClient(app) as client:
        for _ in range(200):
            if instructions:
                break
            client.get("/health")
            time.sleep(0.01)
    assert instructions and MINE in instructions[0]
    assert "하느님" not in instructions[0]
