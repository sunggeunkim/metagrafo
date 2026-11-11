import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from core.event_bus import EventBus
from core.settings import Settings
from features.operator_control.events import CleanupContextEvent
from main import create_app

CONTEXT = "Sermon: the problem of evil and suffering.\nScripture: Job 1-2.\nSpeaker: Pastor Frank (youth)."


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("CLEANUP_CONTEXT", raising=False)
    path = tmp_path / ".env"
    path.write_text("AUDIO_DEVICE_NAME=Booth Mic\n", encoding="utf-8")
    return path


def _app(env_file, **kwargs):
    settings = Settings(caption_engine="whisper", gemini_api_key="", _env_file=env_file)
    return create_app(settings=settings, env_file=env_file, **kwargs)


def test_cleanup_context_starts_empty(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        assert client.get("/mode").json()["cleanup_context"] == ""


def test_saved_context_is_announced_and_kept_beside_env_for_next_time(env_file) -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler(event: CleanupContextEvent) -> None:
        seen.append(event.text)

    bus.subscribe(CleanupContextEvent, handler)
    with TestClient(_app(env_file, bus=bus)) as client:
        response = client.put("/cleanup-context", json={"text": f"  {CONTEXT}\n\n"})
        assert response.status_code == 200
        assert response.json()["cleanup_context"] == CONTEXT
    assert seen == [CONTEXT]
    # Several lines do not fit one line of .env, so the text has its own file; .env is untouched.
    assert (env_file.parent / "cleanup-context.txt").read_text(encoding="utf-8") == CONTEXT
    assert env_file.read_text(encoding="utf-8") == "AUDIO_DEVICE_NAME=Booth Mic\n"
    with TestClient(_app(env_file)) as client:
        assert client.get("/mode").json()["cleanup_context"] == CONTEXT


def test_clearing_the_context_empties_the_file(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        client.put("/cleanup-context", json={"text": CONTEXT})
        response = client.put("/cleanup-context", json={"text": ""})
        assert response.json()["cleanup_context"] == ""
    assert (env_file.parent / "cleanup-context.txt").read_text(encoding="utf-8") == ""


def test_a_context_too_long_for_every_cleanup_call_is_refused(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        response = client.put("/cleanup-context", json={"text": "x" * 2001})
        assert response.status_code == 422
        assert client.get("/mode").json()["cleanup_context"] == ""


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


def test_the_saved_context_reaches_the_cleanup_model(env_file, monkeypatch) -> None:
    from google import genai

    instructions: list[str] = []

    class RecordingClient:
        def __init__(self, **_kwargs: object) -> None:
            async def generate_content(**kw: object) -> SimpleNamespace:
                instructions.append(kw["config"].system_instruction)
                return SimpleNamespace(text="Today we talk about suffering.")

            self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    monkeypatch.setattr(genai, "Client", RecordingClient)
    (env_file.parent / "cleanup-context.txt").write_text(CONTEXT, encoding="utf-8")
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
    assert instructions and CONTEXT in instructions[0]
