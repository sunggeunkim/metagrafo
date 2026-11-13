import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from core.settings import Settings
from features.capture_audio.events import AudioChunkEvent
from features.translate_speech.events import SubtitleEvent
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


class _Booth:
    """One Metagrafo with stand-ins for Gemini, Hermeneia and Whisper on this PC."""

    def __init__(self, tmp_path, monkeypatch, *, env: str = "", load_whisper: bool = False, **settings) -> None:
        for name in ("GEMINI_API_KEY", "CAPTION_ENGINE", "HERMENEIA_URL", "HERMENEIA_TOKEN"):
            monkeypatch.delenv(name, raising=False)
        self.env_file = tmp_path / ".env"
        self.env_file.write_text(env, encoding="utf-8")
        self.opened: list[str] = []
        self.servers: list[tuple[str, str]] = []
        self.captions: list[str] = []

        def make_hermeneia_transcribe(url: str, token: str, _model: str, **_kwargs: object):
            self.servers.append((url, token))
            return lambda _pcm, **_kw: "Hello from Hermeneia."

        monkeypatch.setattr(
            "features.translate_speech.hermeneia_client.make_hermeneia_transcribe", make_hermeneia_transcribe
        )
        monkeypatch.setattr(
            "features.translate_speech.whisper_asr.make_transcribe",
            lambda _profile: lambda _pcm, **_kw: "Hello from this PC.",
        )

        def factory(api_key: str):
            def connect(_mode: object) -> _HoldingSession:
                self.opened.append(api_key)
                return _HoldingSession()

            return connect

        self.app = create_app(
            settings=Settings(_env_file=self.env_file, **settings),
            env_file=self.env_file,
            gemini_connect_factory=factory,
            load_whisper=load_whisper,
        )

        async def collect(event: SubtitleEvent) -> None:
            self.captions.append(event.text)

        self.app.state.bus.subscribe(SubtitleEvent, collect)

    def speak(self, client: TestClient) -> str:
        """Send one cut of speech and return the caption it became."""
        before = len(self.captions)
        chunk = AudioChunkEvent(
            chunk_id=f"c{before}", pcm_s16le=b"\x00\x10" * 1600, sample_rate=16000, started_at=time.monotonic(), duration_s=0.1
        )
        client.portal.call(self.app.state.bus.publish, chunk)
        for _ in range(200):
            if len(self.captions) > before:
                return self.captions[-1]
            client.get("/health")
            time.sleep(0.01)
        raise AssertionError("no caption")

    def saved(self) -> Settings:
        return Settings(_env_file=self.env_file)


def test_hermeneia_is_the_default_server_so_whisper_needs_only_its_token(tmp_path, monkeypatch) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="whisper", hermeneia_token="tok-42", load_whisper=True)
    assert booth.saved().hermeneia_url == "https://hermeneia.stugen.net"
    with TestClient(booth.app) as client:
        assert booth.speak(client) == "Hello from Hermeneia."
    assert booth.servers == [("https://hermeneia.stugen.net", "tok-42")]


def test_whisper_without_a_hermeneia_token_runs_on_this_pc(tmp_path, monkeypatch) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="whisper", load_whisper=True)
    with TestClient(booth.app) as client:
        assert booth.speak(client) == "Hello from this PC."
    assert booth.servers == []


def test_a_saved_hermeneia_token_is_kept_for_next_time_and_never_shown(tmp_path, monkeypatch) -> None:
    booth = _Booth(tmp_path, monkeypatch, env="AUDIO_DEVICE_NAME=Booth Mic\n", caption_engine="whisper")
    with TestClient(booth.app) as client:
        assert client.get("/mode").json()["hermeneia_token_set"] is False
        assert client.put("/hermeneia-token", json={"token": "  "}).status_code == 422
        response = client.put("/hermeneia-token", json={"token": "  tok-42  "})
        assert response.status_code == 200
        assert response.json()["hermeneia_token_set"] is True
        assert "tok-42" not in response.text
        assert "tok-42" not in client.get("/mode").text
    saved = booth.saved()
    assert saved.hermeneia_token == "tok-42"
    assert saved.audio_device_name == "Booth Mic"


def test_a_token_saved_while_whisper_runs_on_this_pc_moves_it_to_hermeneia(tmp_path, monkeypatch) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="whisper", load_whisper=True)
    with TestClient(booth.app) as client:
        assert booth.speak(client) == "Hello from this PC."
        client.put("/hermeneia-token", json={"token": "tok-42"})
        assert booth.speak(client) == "Hello from Hermeneia."
    assert booth.servers == [("https://hermeneia.stugen.net", "tok-42")]


def test_booth_control_switches_from_gemini_to_whisper_on_hermeneia_and_back(tmp_path, monkeypatch) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="gemini_live", gemini_api_key="gem-key")
    with TestClient(booth.app) as client:
        assert client.get("/mode").json()["caption_engine"] == "gemini_live"
        assert booth.opened == ["gem-key"]
        client.put("/hermeneia-token", json={"token": "tok-42"})
        response = client.put("/caption-engine", json={"engine": "whisper"})
        assert response.status_code == 200
        assert response.json()["caption_engine"] == "whisper"
        assert response.json()["gemini_live"] is False
        assert booth.speak(client) == "Hello from Hermeneia."
        assert booth.saved().caption_engine == "whisper"

        response = client.put("/caption-engine", json={"engine": "gemini_live"})
        assert response.status_code == 200
        assert response.json()["caption_engine"] == "gemini_live"
        for _ in range(100):
            if len(booth.opened) == 2:
                break
            client.get("/health")
            time.sleep(0.01)
        assert booth.opened == ["gem-key", "gem-key"]
    assert booth.saved().caption_engine == "gemini_live"


def test_whisper_is_refused_with_no_hermeneia_token_and_no_whisper_on_this_pc(tmp_path, monkeypatch) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="gemini_live", gemini_api_key="gem-key")
    with TestClient(booth.app) as client:
        response = client.put("/caption-engine", json={"engine": "whisper"})
        assert response.status_code == 409
        assert "Hermeneia token" in response.json()["detail"]
        assert client.get("/mode").json()["caption_engine"] == "gemini_live"


def test_whisper_is_refused_while_captions_are_in_korean(tmp_path, monkeypatch) -> None:
    booth = _Booth(
        tmp_path, monkeypatch, caption_engine="gemini_live", gemini_api_key="gem-key", hermeneia_token="tok-42"
    )
    with TestClient(booth.app) as client:
        assert client.put("/mode", json={"mode": "en_to_ko"}).status_code == 200
        response = client.put("/caption-engine", json={"engine": "whisper"})
        assert response.status_code == 409
        assert "Korean" in response.json()["detail"]


def test_gemini_is_refused_without_a_gemini_key(tmp_path, monkeypatch) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="whisper", hermeneia_token="tok-42")
    with TestClient(booth.app) as client:
        response = client.put("/caption-engine", json={"engine": "gemini_live"})
        assert response.status_code == 409
        assert "Gemini key" in response.json()["detail"]
        assert client.get("/mode").json()["caption_engine"] == "whisper"


@pytest.mark.parametrize("engine", ["", "hermeneia"])
def test_an_unknown_engine_is_refused(tmp_path, monkeypatch, engine: str) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="whisper", hermeneia_token="tok-42")
    with TestClient(booth.app) as client:
        assert client.put("/caption-engine", json={"engine": engine}).status_code == 422
