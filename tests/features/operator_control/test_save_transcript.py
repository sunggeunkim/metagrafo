import pytest
from fastapi.testclient import TestClient

from core.event_bus import EventBus
from core.settings import Settings
from features.operator_control.events import SaveTranscriptEvent
from main import create_app


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("SAVE_TRANSCRIPT", raising=False)
    path = tmp_path / ".env"
    path.write_text("AUDIO_DEVICE_NAME=Booth Mic\n", encoding="utf-8")
    return path


def _app(env_file, tmp_path, **kwargs):
    settings = Settings(caption_engine="whisper", gemini_api_key="", _env_file=env_file)
    return create_app(
        settings=settings, env_file=env_file, transcripts_dir=tmp_path / "transcripts", **kwargs
    )


def test_save_transcript_starts_off(env_file, tmp_path) -> None:
    with TestClient(_app(env_file, tmp_path)) as client:
        assert client.get("/mode").json()["save_transcript"] is False


def test_switching_on_tells_the_app_and_keeps_it_for_next_time(env_file, tmp_path) -> None:
    bus = EventBus()
    seen: list[bool] = []

    async def handler(event: SaveTranscriptEvent) -> None:
        seen.append(event.save_transcript)

    bus.subscribe(SaveTranscriptEvent, handler)
    with TestClient(_app(env_file, tmp_path, bus=bus)) as client:
        response = client.put("/save-transcript", json={"save_transcript": True})
        assert response.status_code == 200
        assert response.json()["save_transcript"] is True
        assert client.get("/mode").json()["save_transcript"] is True
    assert seen == [True]
    assert env_file.read_text(encoding="utf-8") == "AUDIO_DEVICE_NAME=Booth Mic\nSAVE_TRANSCRIPT=true\n"
    with TestClient(_app(env_file, tmp_path)) as client:
        assert client.get("/mode").json()["save_transcript"] is True
