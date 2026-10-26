import pytest
from fastapi.testclient import TestClient

from core.settings import Settings
from main import create_app

URL = "http://upload.youtube.com/closedcaption?cid=abc-123"


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("YOUTUBE_CAPTION_URL", raising=False)
    monkeypatch.delenv("AUDIO_DEVICE_NAME", raising=False)
    path = tmp_path / ".env"
    path.write_text("AUDIO_DEVICE_NAME=Booth Mic\nYOUTUBE_CAPTION_URL=\n", encoding="utf-8")
    return path


def _app(env_file):
    settings = Settings(caption_engine="whisper", gemini_api_key="", _env_file=env_file)
    return create_app(settings=settings, env_file=env_file)


def test_saving_a_url_starts_sending_and_keeps_it_for_next_time(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        assert client.get("/mode").json()["youtube_url_set"] is False
        assert client.get("/youtube-captions").json() == {"enabled": False}

        response = client.put("/youtube-caption-url", json={"url": f"  {URL}  "})
        assert response.status_code == 200
        assert response.json()["youtube_url_set"] is True
        assert "abc-123" not in response.text
        assert client.get("/youtube-captions").json()["enabled"] is True

    reloaded = Settings(_env_file=env_file)
    assert reloaded.youtube_caption_url == URL
    assert reloaded.audio_device_name == "Booth Mic"
    assert env_file.read_text(encoding="utf-8").count("YOUTUBE_CAPTION_URL=") == 1


def test_stop_sending_clears_the_saved_url(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        client.put("/youtube-caption-url", json={"url": URL})
        response = client.put("/youtube-caption-url", json={"url": ""})
        assert response.status_code == 200
        assert response.json()["youtube_url_set"] is False
        assert client.get("/youtube-captions").json() == {"enabled": False}

    assert Settings(_env_file=env_file).youtube_caption_url == ""


@pytest.mark.parametrize(
    "bad",
    ["not a url", "ftp://upload.youtube.com/closedcaption", f"{URL}\nGEMINI_API_KEY=x", "http://"],
)
def test_rejects_something_that_is_not_a_caption_url(env_file, bad: str) -> None:
    with TestClient(_app(env_file)) as client:
        response = client.put("/youtube-caption-url", json={"url": bad})
        assert response.status_code == 422
        assert client.get("/youtube-captions").json() == {"enabled": False}

    assert Settings(_env_file=env_file).youtube_caption_url == ""
