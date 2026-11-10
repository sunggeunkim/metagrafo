import pytest
from fastapi.testclient import TestClient

from core.event_bus import EventBus
from core.settings import Settings
from features.operator_control.events import CleanUpWordingEvent
from main import create_app


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("CLEAN_UP_WORDING", raising=False)
    monkeypatch.delenv("CLEANUP_MODEL", raising=False)
    path = tmp_path / ".env"
    path.write_text("AUDIO_DEVICE_NAME=Booth Mic\n", encoding="utf-8")
    return path


def _app(env_file, **kwargs):
    settings = Settings(caption_engine="whisper", gemini_api_key="", _env_file=env_file)
    return create_app(settings=settings, env_file=env_file, **kwargs)


def test_clean_up_wording_starts_on(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        assert client.get("/health").json()["clean_up_wording"] is True
        assert client.get("/mode").json()["clean_up_wording"] is True


def test_switching_off_tells_gemini_and_keeps_it_for_next_time(env_file) -> None:
    bus = EventBus()
    seen: list[bool] = []

    async def handler(event: CleanUpWordingEvent) -> None:
        seen.append(event.clean_up_wording)

    bus.subscribe(CleanUpWordingEvent, handler)
    with TestClient(_app(env_file, bus=bus)) as client:
        response = client.put("/clean-up-wording", json={"clean_up_wording": False})
        assert response.status_code == 200
        assert response.json()["clean_up_wording"] is False
        assert client.get("/health").json()["clean_up_wording"] is False
    assert seen == [False]
    assert env_file.read_text(encoding="utf-8") == "AUDIO_DEVICE_NAME=Booth Mic\nCLEAN_UP_WORDING=false\n"
    with TestClient(_app(env_file)) as client:
        assert client.get("/health").json()["clean_up_wording"] is False
