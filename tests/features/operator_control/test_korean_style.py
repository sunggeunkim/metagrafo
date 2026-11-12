import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from core.event_bus import EventBus
from core.settings import KOREAN_STYLE_DEFAULT, Settings
from features.operator_control.events import KoreanStyleEvent
from main import create_app

YOUTH = "청소년 예배입니다. 해요체로 친근하게 쓰되, 하나님께는 높임말을 쓰십시오."


@pytest.fixture
def env_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text("AUDIO_DEVICE_NAME=Booth Mic\n", encoding="utf-8")
    return path


def _app(env_file, **kwargs):
    settings = Settings(caption_engine="whisper", gemini_api_key="", _env_file=env_file)
    return create_app(settings=settings, env_file=env_file, **kwargs)


def _style_file(env_file):
    return env_file.parent / "korean-style.txt"


def test_korean_style_starts_as_the_formal_sermon_default(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        body = client.get("/mode").json()
    assert body["korean_style"] == KOREAN_STYLE_DEFAULT
    assert body["korean_style_default"] == KOREAN_STYLE_DEFAULT


def test_a_saved_style_is_announced_and_kept_beside_env_for_next_time(env_file) -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler(event: KoreanStyleEvent) -> None:
        seen.append(event.text)

    bus.subscribe(KoreanStyleEvent, handler)
    with TestClient(_app(env_file, bus=bus)) as client:
        response = client.put("/korean-style", json={"text": f"  {YOUTH}\n"})
        assert response.status_code == 200
        assert response.json()["korean_style"] == YOUTH
    assert seen == [YOUTH]
    assert _style_file(env_file).read_text(encoding="utf-8") == YOUTH
    assert env_file.read_text(encoding="utf-8") == "AUDIO_DEVICE_NAME=Booth Mic\n"
    with TestClient(_app(env_file)) as client:
        assert client.get("/mode").json()["korean_style"] == YOUTH


def test_saving_an_empty_style_turns_the_rule_off_and_it_stays_off(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        assert client.put("/korean-style", json={"text": "  "}).json()["korean_style"] == ""
    with TestClient(_app(env_file)) as client:
        assert client.get("/mode").json()["korean_style"] == ""


def test_reset_goes_back_to_the_default_and_stays_there(env_file) -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler(event: KoreanStyleEvent) -> None:
        seen.append(event.text)

    bus.subscribe(KoreanStyleEvent, handler)
    with TestClient(_app(env_file, bus=bus)) as client:
        client.put("/korean-style", json={"text": YOUTH})
        response = client.delete("/korean-style")
        assert response.status_code == 200
        assert response.json()["korean_style"] == KOREAN_STYLE_DEFAULT
    assert seen == [YOUTH, KOREAN_STYLE_DEFAULT]
    assert not _style_file(env_file).exists()
    with TestClient(_app(env_file)) as client:
        assert client.get("/mode").json()["korean_style"] == KOREAN_STYLE_DEFAULT


def test_a_style_too_long_for_every_cleanup_call_is_refused(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        assert client.put("/korean-style", json={"text": "x" * 1001}).status_code == 422
        assert client.get("/mode").json()["korean_style"] == KOREAN_STYLE_DEFAULT


def test_the_file_is_the_only_source_so_an_env_value_is_ignored(env_file, monkeypatch) -> None:
    monkeypatch.setenv("KOREAN_STYLE", "From the shell, never saved on booth control.")
    with TestClient(_app(env_file)) as client:
        assert client.get("/mode").json()["korean_style"] == KOREAN_STYLE_DEFAULT


def test_a_hand_edited_file_over_the_limit_is_cut_to_it(env_file, caplog) -> None:
    _style_file(env_file).write_text("x" * 5000, encoding="utf-8")
    with TestClient(_app(env_file)) as client:
        assert client.get("/mode").json()["korean_style"] == "x" * 1000
    assert "korean-style.txt" in caplog.text


def test_an_unreadable_file_falls_back_to_the_default(env_file, caplog) -> None:
    _style_file(env_file).write_bytes(b"\xff\xfe\x00 not utf-8 \x81")
    with TestClient(_app(env_file)) as client:
        assert client.get("/mode").json()["korean_style"] == KOREAN_STYLE_DEFAULT
    assert "korean-style.txt" in caplog.text


class _SpeakingSession:
    """A Gemini socket that says one Korean sentence, then stays open: the Korean style is only
    for Korean captions."""

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
                output_transcription=SimpleNamespace(text="오늘은 고통의 문제에 대해 여러분과 함께 이야기하겠습니다."),
                turn_complete=False,
            ),
        )
        await asyncio.Event().wait()


def test_the_saved_style_reaches_the_cleanup_model(env_file, monkeypatch) -> None:
    from google import genai

    instructions: list[str] = []

    class RecordingClient:
        def __init__(self, **_kwargs: object) -> None:
            async def generate_content(**kw: object) -> SimpleNamespace:
                instructions.append(kw["config"].system_instruction)
                return SimpleNamespace(text="Today we talk about suffering.")

            self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    monkeypatch.setattr(genai, "Client", RecordingClient)
    _style_file(env_file).write_text(YOUTH, encoding="utf-8")
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
    assert instructions and YOUTH in instructions[0]
    assert "never 해요체 or 반말" not in instructions[0]
