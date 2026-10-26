import asyncio

from fastapi.testclient import TestClient

from core.settings import Settings
from main import create_app


class _HoldingSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def send_realtime_input(self, **_kwargs: object) -> None:
        return None

    async def receive(self):
        await asyncio.Event().wait()
        yield None


def test_booth_opens_when_whisper_is_not_installed(monkeypatch) -> None:
    def missing(_profile: object) -> str:
        raise ImportError("no whisper")

    monkeypatch.setattr(
        "features.translate_speech.whisper_asr.make_transcribe",
        missing,
    )
    settings = Settings(caption_engine="whisper", gemini_api_key="")
    app = create_app(settings=settings, load_whisper=True)
    with TestClient(app) as client:
        assert client.get("/control").status_code == 200
        assert client.get("/health").json()["ok"] is True


def test_pasting_a_gemini_key_connects_with_that_key_and_keeps_other_settings(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("CAPTION_ENGINE", raising=False)
    monkeypatch.delenv("AUDIO_DEVICE_NAME", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("AUDIO_DEVICE_NAME=Booth Mic\n", encoding="utf-8")
    opened: list[str] = []

    def factory(api_key: str):
        def connect(_mode: object) -> _HoldingSession:
            opened.append(api_key)
            return _HoldingSession()

        return connect

    settings = Settings(
        caption_engine="whisper",
        gemini_api_key="",
        _env_file=env_file,
    )
    app = create_app(settings=settings, env_file=env_file, gemini_connect_factory=factory)
    with TestClient(app) as client:
        assert client.get("/mode").json()["gemini_key_set"] is False
        response = client.put("/gemini-key", json={"api_key": "  pasted-key-42  "})
        assert response.status_code == 200
        assert response.json()["gemini_key_set"] is True
        assert "pasted-key-42" not in response.text
        assert opened == ["pasted-key-42"]

        response = client.put("/gemini-key", json={"api_key": "replacement-key"})
        assert response.status_code == 200
        assert "replacement-key" not in response.text
        assert opened == ["pasted-key-42", "replacement-key"]

    reloaded = Settings(_env_file=env_file)
    assert reloaded.gemini_live() is True
    assert reloaded.gemini_api_key == "replacement-key"
    assert reloaded.audio_device_name == "Booth Mic"


def test_windowless_start_logs_to_a_file_so_uvicorn_can_configure(tmp_path, monkeypatch) -> None:
    import sys

    from uvicorn.config import Config

    from main import _redirect_missing_stdio

    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    log_path = tmp_path / "metagrafo.log"
    _redirect_missing_stdio(log_path)
    Config(app="main:app").configure_logging()
    print("IN: hello", flush=True)
    sys.stdout.close()
    assert "IN: hello" in log_path.read_text(encoding="utf-8")


def test_saved_key_unlocks_korean_captions(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("CAPTION_ENGINE", raising=False)
    env_file = tmp_path / ".env"
    targets: list[object] = []

    def factory(_api_key: str):
        def connect(mode: object) -> _HoldingSession:
            targets.append(mode)
            return _HoldingSession()

        return connect

    settings = Settings(caption_engine="whisper", gemini_api_key="", _env_file=env_file)
    app = create_app(settings=settings, env_file=env_file, gemini_connect_factory=factory)
    with TestClient(app) as client:
        assert client.put("/mode", json={"mode": "en_to_ko"}).status_code == 409
        client.put("/gemini-key", json={"api_key": "pasted-key-42"})
        assert client.get("/mode").json()["gemini_live"] is True
        response = client.put("/mode", json={"mode": "en_to_ko"})
        assert response.status_code == 200
        assert response.json()["source_language"] == "en"
        assert response.json()["target_language"] == "ko"


def test_a_key_pasted_after_a_whisper_fallback_keeps_the_booth_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("CAPTION_ENGINE", raising=False)
    env_file = tmp_path / ".env"
    modes: list[str] = []

    def factory(_api_key: str):
        def connect(mode: object) -> _HoldingSession:
            modes.append(str(mode))
            return _HoldingSession()

        return connect

    settings = Settings(
        caption_engine="whisper",
        gemini_api_key="",
        translate_mode="en_to_ko",
        _env_file=env_file,
    )
    app = create_app(settings=settings, env_file=env_file, gemini_connect_factory=factory)
    with TestClient(app) as client:
        assert client.get("/mode").json()["mode"] == "ko_to_en"
        client.put("/gemini-key", json={"api_key": "pasted-key-42"})
        assert client.get("/mode").json()["mode"] == "ko_to_en"
        assert modes == ["ko_to_en"]


def test_pasting_a_key_starts_gemini_in_the_mode_the_booth_shows(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("CAPTION_ENGINE", raising=False)
    env_file = tmp_path / ".env"
    modes: list[str] = []

    def factory(_api_key: str):
        def connect(mode: object) -> _HoldingSession:
            modes.append(str(mode))
            return _HoldingSession()

        return connect

    settings = Settings(
        caption_engine="whisper",
        gemini_api_key="",
        translate_mode="ko_to_en",
        _env_file=env_file,
    )
    app = create_app(settings=settings, env_file=env_file, gemini_connect_factory=factory)
    with TestClient(app) as client:
        assert client.put("/mode", json={"mode": "en_to_en"}).status_code == 200
        client.put("/gemini-key", json={"api_key": "pasted-key-42"})
        assert client.get("/mode").json()["mode"] == "en_to_en"
        assert modes == ["en_to_en"]
