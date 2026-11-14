import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

from core.settings import Settings
from features.capture_audio.events import AudioChunkEvent
from features.translate_speech.events import SubtitleEvent
from main import create_app


class _HoldingSession:
    def __init__(self, close_s: float = 0.0) -> None:
        self._close_s = close_s

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args: object) -> bool:
        # Closing a real socket takes a while; a switch must not race past it.
        await asyncio.sleep(self._close_s)
        return False

    async def send_realtime_input(self, **_kwargs: object) -> None:
        return None

    async def receive(self):
        await asyncio.Event().wait()
        yield None


class _Booth:
    """One Metagrafo with stand-ins for Gemini, Hermeneia and Whisper on this PC."""

    def __init__(
        self,
        tmp_path,
        monkeypatch,
        *,
        env: str = "",
        load_whisper: bool = False,
        close_s: float = 0.0,
        whisper_installed: bool = True,
        whisper_fails: bool = False,
        load_gate: threading.Event | None = None,
        **settings,
    ) -> None:
        for name in (
            "GEMINI_API_KEY", "CAPTION_ENGINE", "HERMENEIA_URL", "HERMENEIA_TOKEN", "HERMENEIA_MODEL", "USE_HERMENEIA"
        ):
            monkeypatch.delenv(name, raising=False)
        self.env_file = tmp_path / ".env"
        self.env_file.write_text(env, encoding="utf-8")
        self.opened: list[str] = []
        self.servers: list[tuple[str, str]] = []
        self.models: list[str] = []
        self.loads = 0
        self.captions: list[str] = []
        self.subtitles: list[SubtitleEvent] = []
        # Install Whisper can make this true while Metagrafo runs.
        self.installed = whisper_installed

        def make_hermeneia_transcribe(url: str, token: str, model: str, **_kwargs: object):
            self.servers.append((url, token))
            self.models.append(model)
            name = "Seamless" if model == "seamless-m4t-v2" else "Whisper"

            def transcribe(_pcm, **kwargs):
                # The client sends the target: Seamless writes Korean when asked, Whisper never.
                if name == "Seamless" and kwargs.get("target_language") == "ko":
                    return "안녕하세요, 하나님의 말씀입니다."
                return f"Hello from {name} on Hermeneia."

            return transcribe

        def make_transcribe(_profile):
            if not self.installed:
                raise ImportError("No module named 'faster_whisper'")
            self.loads += 1
            if load_gate is not None:
                # Loading onto the GPU takes a while; other clicks land meanwhile.
                assert load_gate.wait(5)
            if whisper_fails:
                raise RuntimeError("CUDA failed with error out of memory")
            return lambda _pcm, **_kw: "Hello from this PC."

        monkeypatch.setattr(
            "features.translate_speech.hermeneia_client.make_hermeneia_transcribe", make_hermeneia_transcribe
        )
        monkeypatch.setattr("features.translate_speech.whisper_asr.make_transcribe", make_transcribe)
        monkeypatch.setattr("features.translate_speech.whisper_asr.installed", lambda: self.installed)

        def factory(api_key: str):
            def connect(_mode: object) -> _HoldingSession:
                self.opened.append(api_key)
                return _HoldingSession(close_s)

            return connect

        self.app = create_app(
            settings=Settings(_env_file=self.env_file, **settings),
            env_file=self.env_file,
            gemini_connect_factory=factory,
            load_whisper=load_whisper,
        )

        async def collect(event: SubtitleEvent) -> None:
            self.captions.append(event.text)
            self.subtitles.append(event)

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

    def wait_until(self, client: TestClient, done) -> None:
        for _ in range(200):
            if done():
                return
            client.get("/health")
            time.sleep(0.01)
        raise AssertionError("timed out")

    def put_while_loading(self, client: TestClient, engine: str):
        """PUT /caption-engine from another tab; returns the thread and where its response lands."""
        responses: list = []
        thread = threading.Thread(
            target=lambda: responses.append(client.put("/caption-engine", json={"engine": engine}))
        )
        thread.start()
        self.wait_until(client, lambda: self.loads == 1)
        return thread, responses

    def saved(self) -> Settings:
        return Settings(_env_file=self.env_file)


def test_hermeneia_is_the_default_server_so_whisper_needs_only_its_token(tmp_path, monkeypatch) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="whisper", hermeneia_token="tok-42", load_whisper=True)
    assert booth.saved().hermeneia_url == "https://hermeneia.stugen.net"
    with TestClient(booth.app) as client:
        assert booth.speak(client) == "Hello from Whisper on Hermeneia."
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
        assert booth.speak(client) == "Hello from Whisper on Hermeneia."
    assert booth.servers == [("https://hermeneia.stugen.net", "tok-42")]


def test_booth_control_switches_from_gemini_to_whisper_on_hermeneia_and_back(tmp_path, monkeypatch) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="gemini_live", gemini_api_key="gem-key")
    with TestClient(booth.app) as client:
        assert client.get("/mode").json()["caption_engine"] == "gemini_live"
        assert booth.opened == ["gem-key"]
        client.put("/hermeneia-token", json={"token": "tok-42"})
        response = client.put("/caption-engine", json={"engine": "whisper_hermeneia"})
        assert response.status_code == 200
        assert response.json()["caption_engine"] == "whisper_hermeneia"
        assert response.json()["gemini_live"] is False
        assert booth.speak(client) == "Hello from Whisper on Hermeneia."
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
        response = client.put("/caption-engine", json={"engine": "whisper_hermeneia"})
        assert response.status_code == 409
        assert "Hermeneia token" in response.json()["detail"]
        assert client.get("/mode").json()["caption_engine"] == "gemini_live"


def test_whisper_is_refused_while_captions_are_in_korean(tmp_path, monkeypatch) -> None:
    booth = _Booth(
        tmp_path, monkeypatch, caption_engine="gemini_live", gemini_api_key="gem-key", hermeneia_token="tok-42"
    )
    with TestClient(booth.app) as client:
        assert client.put("/mode", json={"mode": "en_to_ko"}).status_code == 200
        response = client.put("/caption-engine", json={"engine": "whisper_hermeneia"})
        assert response.status_code == 409
        assert "Korean" in response.json()["detail"]


def test_gemini_is_refused_without_a_gemini_key(tmp_path, monkeypatch) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="whisper", hermeneia_token="tok-42")
    with TestClient(booth.app) as client:
        response = client.put("/caption-engine", json={"engine": "gemini_live"})
        assert response.status_code == 409
        assert "Gemini key" in response.json()["detail"]
        assert client.get("/mode").json()["caption_engine"] == "whisper_hermeneia"


@pytest.mark.parametrize("engine", ["", "hermeneia", "whisper", "seamless_local"])
def test_an_unknown_engine_is_refused(tmp_path, monkeypatch, engine: str) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="whisper", hermeneia_token="tok-42")
    with TestClient(booth.app) as client:
        assert client.put("/caption-engine", json={"engine": engine}).status_code == 422


def test_a_line_gemini_had_open_never_shows_after_switching_to_whisper_or_back(tmp_path, monkeypatch) -> None:
    booth = _Booth(
        tmp_path,
        monkeypatch,
        caption_engine="gemini_live",
        gemini_api_key="gem-key",
        hermeneia_token="tok-42",
        whole_sentences_quiet_s=0.1,
        clean_up_wording=False,
    )
    with TestClient(booth.app) as client:
        live = booth.app.state.translator.live
        # Gemini has half a sentence open when the booth switches to Whisper.
        client.portal.call(live.apply_transcript, "And so we come before")
        client.put("/caption-engine", json={"engine": "whisper_hermeneia"})
        time.sleep(0.3)
        client.get("/health")
        # Its quiet wait must not show it once Whisper runs.
        assert booth.captions == []
        client.put("/caption-engine", json={"engine": "gemini_live"})
        booth.wait_until(client, lambda: len(booth.opened) == 2)
        client.portal.call(live.apply_transcript, "we pray for the church in Rome today.")
        # Back on Gemini, the new sentence starts afresh, with nothing of the old line.
        assert booth.captions == ["we pray for the church in Rome today."]


def test_overlapping_switches_end_on_the_engine_picked_last(tmp_path, monkeypatch) -> None:
    from features.operator_control.events import CaptionEngineChangedEvent

    booth = _Booth(
        tmp_path,
        monkeypatch,
        caption_engine="gemini_live",
        gemini_api_key="gem-key",
        hermeneia_token="tok-42",
        close_s=0.1,
    )
    settings = booth.app.state.settings
    bus = booth.app.state.bus

    async def whisper_then_gemini() -> None:
        # A click on Whisper, then on Gemini while the Gemini socket is still closing.
        settings.caption_engine = "whisper"
        first = asyncio.create_task(bus.publish(CaptionEngineChangedEvent()))
        await asyncio.sleep(0.02)
        settings.caption_engine = "gemini_live"
        second = asyncio.create_task(bus.publish(CaptionEngineChangedEvent()))
        await asyncio.gather(first, second)

    with TestClient(booth.app) as client:
        client.portal.call(whisper_then_gemini)
        booth.wait_until(client, lambda: len(booth.opened) == 2)
        chunk = AudioChunkEvent(chunk_id="late", pcm_s16le=b"\x00\x10" * 1600, sample_rate=16000, started_at=time.monotonic(), duration_s=0.1)
        client.portal.call(bus.publish, chunk)
        time.sleep(0.2)
        client.get("/health")
    # Gemini runs again, and Whisper, which lost, captions nothing.
    assert booth.opened == ["gem-key", "gem-key"]
    assert booth.captions == []


def test_gemini_starts_from_whisper_with_a_key_saved_before_this_start(tmp_path, monkeypatch) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="whisper", gemini_api_key="gem-key", hermeneia_token="tok-42")
    with TestClient(booth.app) as client:
        assert booth.opened == []
        assert client.put("/caption-engine", json={"engine": "gemini_live"}).status_code == 200
        booth.wait_until(client, lambda: booth.opened == ["gem-key"])


def test_seamless_runs_on_hermeneia_and_is_kept_for_next_time(tmp_path, monkeypatch) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="whisper", hermeneia_token="tok-42")
    with TestClient(booth.app) as client:
        assert booth.speak(client) == "Hello from Whisper on Hermeneia."
        response = client.put("/caption-engine", json={"engine": "seamless_hermeneia"})
        assert response.status_code == 200
        assert response.json()["caption_engine"] == "seamless_hermeneia"
        assert booth.speak(client) == "Hello from Seamless on Hermeneia."
        client.put("/caption-engine", json={"engine": "whisper_hermeneia"})
        assert booth.speak(client) == "Hello from Whisper on Hermeneia."
    assert booth.models == ["whisper", "seamless-m4t-v2", "whisper"]
    saved = booth.saved()
    assert (saved.caption_engine, saved.hermeneia_model) == ("whisper", "whisper")


def test_metagrafo_starts_on_seamless_when_it_was_picked_last_time(tmp_path, monkeypatch) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="whisper", hermeneia_token="tok-42")
    with TestClient(booth.app) as client:
        client.put("/caption-engine", json={"engine": "seamless_hermeneia"})
    again = _Booth(tmp_path, monkeypatch, env=booth.env_file.read_text(encoding="utf-8"), hermeneia_token="tok-42")
    with TestClient(again.app) as client:
        assert client.get("/mode").json()["caption_engine"] == "seamless_hermeneia"
        assert again.speak(client) == "Hello from Seamless on Hermeneia."


def test_whisper_on_this_pc_loads_when_first_picked_even_with_a_hermeneia_token(tmp_path, monkeypatch) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="whisper", hermeneia_token="tok-42", load_whisper=True)
    with TestClient(booth.app) as client:
        # Started on Hermeneia, Whisper is not loaded here: it would only take the GPU from OBS.
        assert booth.loads == 0
        assert client.get("/mode").json()["whisper_local_ready"] is True
        assert booth.speak(client) == "Hello from Whisper on Hermeneia."
        response = client.put("/caption-engine", json={"engine": "whisper_local"})
        assert response.status_code == 200
        assert response.json()["caption_engine"] == "whisper_local"
        assert booth.speak(client) == "Hello from this PC."
        client.put("/caption-engine", json={"engine": "whisper_hermeneia"})
        assert booth.speak(client) == "Hello from Whisper on Hermeneia."
        client.put("/caption-engine", json={"engine": "whisper_local"})
        assert booth.speak(client) == "Hello from this PC."
    # Loaded once, then kept.
    assert booth.loads == 1


def test_whisper_on_this_pc_is_kept_for_next_time_with_a_token_saved(tmp_path, monkeypatch) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="whisper", hermeneia_token="tok-42", load_whisper=True)
    with TestClient(booth.app) as client:
        client.put("/caption-engine", json={"engine": "whisper_local"})
    again = _Booth(
        tmp_path, monkeypatch, env=booth.env_file.read_text(encoding="utf-8"), hermeneia_token="tok-42", load_whisper=True
    )
    with TestClient(again.app) as client:
        assert client.get("/mode").json()["caption_engine"] == "whisper_local"
        assert again.speak(client) == "Hello from this PC."
        # A token saved again does not move a booth that picked this PC.
        client.put("/hermeneia-token", json={"token": "tok-43"})
        assert again.speak(client) == "Hello from this PC."
    assert again.servers == []


def test_whisper_on_this_pc_is_refused_when_whisper_is_not_installed(tmp_path, monkeypatch) -> None:
    booth = _Booth(
        tmp_path,
        monkeypatch,
        caption_engine="whisper",
        hermeneia_token="tok-42",
        load_whisper=True,
        whisper_installed=False,
    )
    with TestClient(booth.app) as client:
        assert client.get("/mode").json()["whisper_local_ready"] is False
        response = client.put("/caption-engine", json={"engine": "whisper_local"})
        assert response.status_code == 409
        assert "not installed" in response.json()["detail"]
        assert booth.speak(client) == "Hello from Whisper on Hermeneia."
    assert booth.saved().use_hermeneia is True


def test_whisper_that_cannot_load_on_this_pc_leaves_hermeneia_running(tmp_path, monkeypatch) -> None:
    booth = _Booth(
        tmp_path,
        monkeypatch,
        caption_engine="whisper",
        hermeneia_token="tok-42",
        load_whisper=True,
        whisper_fails=True,
    )
    with TestClient(booth.app) as client:
        response = client.put("/caption-engine", json={"engine": "whisper_local"})
        assert response.status_code == 409
        assert "could not load" in response.json()["detail"]
        assert client.get("/mode").json()["caption_engine"] == "whisper_hermeneia"
        assert booth.speak(client) == "Hello from Whisper on Hermeneia."
    assert booth.saved().use_hermeneia is True


@pytest.mark.parametrize("engine", ["whisper_hermeneia", "seamless_hermeneia"])
def test_hermeneia_engines_are_refused_without_a_token(tmp_path, monkeypatch, engine: str) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="whisper", load_whisper=True)
    with TestClient(booth.app) as client:
        response = client.put("/caption-engine", json={"engine": engine})
        assert response.status_code == 409
        assert "Hermeneia token" in response.json()["detail"]
        assert client.get("/mode").json()["caption_engine"] == "whisper_local"


def test_whisper_installed_after_a_start_without_it_captions_when_picked(tmp_path, monkeypatch) -> None:
    # Started on Whisper on this PC with no Whisper and no token: the worker has nothing to run.
    booth = _Booth(tmp_path, monkeypatch, caption_engine="whisper", load_whisper=True, whisper_installed=False)
    with TestClient(booth.app) as client:
        assert client.get("/mode").json()["whisper_local_ready"] is False
        booth.installed = True
        assert client.get("/mode").json()["whisper_local_ready"] is True
        response = client.put("/caption-engine", json={"engine": "whisper_local"})
        assert response.status_code == 200
        assert booth.speak(client) == "Hello from this PC."
    assert booth.loads == 1


def test_a_pick_made_while_whisper_loads_on_this_pc_wins(tmp_path, monkeypatch) -> None:
    gate = threading.Event()
    booth = _Booth(
        tmp_path, monkeypatch, caption_engine="whisper", hermeneia_token="tok-42", load_whisper=True, load_gate=gate
    )
    with TestClient(booth.app) as client:
        slow, responses = booth.put_while_loading(client, "whisper_local")
        # Another tab picks Seamless while Whisper is still loading here.
        assert client.put("/caption-engine", json={"engine": "seamless_hermeneia"}).status_code == 200
        gate.set()
        slow.join(5)
        assert responses[0].status_code == 409
        assert "Seamless on Hermeneia" in responses[0].json()["detail"]
        assert client.get("/mode").json()["caption_engine"] == "seamless_hermeneia"
        assert booth.speak(client) == "Hello from Seamless on Hermeneia."
    saved = booth.saved()
    assert (saved.use_hermeneia, saved.hermeneia_model) == (True, "seamless-m4t-v2")


def test_whisper_on_this_pc_is_refused_when_korean_captions_start_while_it_loads(tmp_path, monkeypatch) -> None:
    gate = threading.Event()
    booth = _Booth(
        tmp_path,
        monkeypatch,
        caption_engine="gemini_live",
        gemini_api_key="gem-key",
        load_whisper=True,
        load_gate=gate,
    )
    with TestClient(booth.app) as client:
        slow, responses = booth.put_while_loading(client, "whisper_local")
        # Korean captions are allowed: Gemini is still the engine.
        assert client.put("/mode", json={"mode": "en_to_ko"}).status_code == 200
        gate.set()
        slow.join(5)
        assert responses[0].status_code == 409
        assert "Korean" in responses[0].json()["detail"]
        assert client.get("/mode").json()["caption_engine"] == "gemini_live"
    # .env is left as it was: nothing saved Whisper.
    assert booth.env_file.read_text(encoding="utf-8") == ""


def test_seamless_on_hermeneia_writes_korean_captions_in_en_to_ko(tmp_path, monkeypatch) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="whisper", hermeneia_token="tok-42")
    with TestClient(booth.app) as client:
        # Whisper on Hermeneia only writes English.
        assert client.get("/mode").json()["writes_korean"] is False
        assert client.put("/mode", json={"mode": "en_to_ko"}).status_code == 409
        assert client.put("/caption-engine", json={"engine": "seamless_hermeneia"}).status_code == 200
        assert client.get("/mode").json()["writes_korean"] is True
        response = client.put("/mode", json={"mode": "en_to_ko"})
        assert response.status_code == 200
        assert booth.speak(client) == "안녕하세요, 하나님의 말씀입니다."
    last = booth.subtitles[-1]
    assert (last.mode, last.source_language, last.target_language) == ("en_to_ko", "en", "ko")


def test_seamless_keeps_korean_captions_when_picked_from_gemini_in_en_to_ko(tmp_path, monkeypatch) -> None:
    booth = _Booth(
        tmp_path, monkeypatch, caption_engine="gemini_live", gemini_api_key="gem-key", hermeneia_token="tok-42"
    )
    with TestClient(booth.app) as client:
        assert client.put("/mode", json={"mode": "en_to_ko"}).status_code == 200
        assert client.put("/caption-engine", json={"engine": "seamless_hermeneia"}).status_code == 200
        assert client.get("/mode").json()["mode"] == "en_to_ko"
        assert booth.speak(client) == "안녕하세요, 하나님의 말씀입니다."


@pytest.mark.parametrize("engine", ["whisper_hermeneia", "whisper_local"])
def test_whisper_is_refused_while_seamless_writes_korean(tmp_path, monkeypatch, engine: str) -> None:
    booth = _Booth(tmp_path, monkeypatch, caption_engine="whisper", hermeneia_token="tok-42", load_whisper=True)
    with TestClient(booth.app) as client:
        client.put("/caption-engine", json={"engine": "seamless_hermeneia"})
        client.put("/mode", json={"mode": "en_to_ko"})
        response = client.put("/caption-engine", json={"engine": engine})
        assert response.status_code == 409
        assert "Korean" in response.json()["detail"]
        assert client.get("/mode").json()["caption_engine"] == "seamless_hermeneia"
        assert booth.speak(client) == "안녕하세요, 하나님의 말씀입니다."
    # Whisper on this PC never loaded for a pick that was refused.
    assert booth.loads == 0
