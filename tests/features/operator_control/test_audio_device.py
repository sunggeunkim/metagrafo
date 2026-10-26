import pytest
from fastapi.testclient import TestClient

from core.event_bus import EventBus
from core.settings import Settings
from features.operator_control.events import AudioDeviceChangedEvent
from main import create_app

INPUTS = ["HDMI (Elgato 4K S)", "Realtek Mic"]
# A Windows Bluetooth hands-free name, as PortAudio reports it.
BLUETOOTH = "Headset (@System32\\drivers\\bthhfenum.sys,#2;%1 Hands-Free%0\r\n;(SK200))"


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIO_DEVICE_NAME", raising=False)
    monkeypatch.delenv("AUDIO_DEVICE_INDEX", raising=False)
    path = tmp_path / ".env"
    path.write_text("AUDIO_DEVICE_NAME=HDMI (Elgato 4K S)\nVAD_PAUSES=2\n", encoding="utf-8")
    return path


def _app(env_file, *, bus=None, **overrides):
    settings = Settings(caption_engine="whisper", gemini_api_key="", _env_file=env_file, **overrides)
    return create_app(
        bus=bus, settings=settings, env_file=env_file, list_input_devices=lambda: [*INPUTS, BLUETOOTH]
    )


def test_lists_the_inputs_and_the_one_in_use_but_not_names_with_a_line_break(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        assert client.get("/audio-devices").json() == {
            "devices": INPUTS,
            "selected": "HDMI (Elgato 4K S)",
            "capturing": None,
            "error": None,
        }
        assert client.get("/mode").json()["audio_device"] == "HDMI (Elgato 4K S)"


def test_picking_an_input_switches_capture_and_keeps_it_for_next_time(env_file) -> None:
    bus = EventBus()
    seen: list[AudioDeviceChangedEvent] = []

    async def record(event: AudioDeviceChangedEvent) -> None:
        seen.append(event)

    bus.subscribe(AudioDeviceChangedEvent, record)
    with TestClient(_app(env_file, bus=bus)) as client:
        response = client.put("/audio-device", json={"name": "  Realtek Mic "})
        assert response.status_code == 200
        assert response.json()["devices"] == INPUTS
        assert response.json()["selected"] == "Realtek Mic"
        assert client.get("/mode").json()["audio_device"] == "Realtek Mic"

    assert len(seen) == 1
    reloaded = Settings(_env_file=env_file)
    assert reloaded.audio_device_name == "Realtek Mic"
    assert reloaded.vad_pauses == 2
    assert env_file.read_text(encoding="utf-8").count("AUDIO_DEVICE_NAME=") == 1


@pytest.mark.parametrize("bad", ["", "   ", "Speakers", "Realtek Mic\nGEMINI_API_KEY=x", BLUETOOTH])
def test_rejects_a_name_that_is_not_a_listed_input(env_file, bad: str) -> None:
    before = env_file.read_text(encoding="utf-8")
    with TestClient(_app(env_file)) as client:
        response = client.put("/audio-device", json={"name": bad})
        assert response.status_code == 422
        assert client.get("/audio-devices").json()["selected"] == "HDMI (Elgato 4K S)"

    assert env_file.read_text(encoding="utf-8") == before


def test_refuses_while_an_index_overrides_the_name(env_file) -> None:
    with TestClient(_app(env_file, audio_device_index=3)) as client:
        response = client.put("/audio-device", json={"name": "Realtek Mic"})
        assert response.status_code == 409
        assert "AUDIO_DEVICE_INDEX" in response.json()["detail"]

    assert Settings(_env_file=env_file).audio_device_name == "HDMI (Elgato 4K S)"


def test_shows_what_capture_opened_or_why_it_could_not(env_file) -> None:
    from fastapi import FastAPI

    from features import operator_control

    status = {"capturing": None, "error": "[Errno -9999] Unanticipated host error"}
    app = FastAPI()
    operator_control.register(
        app,
        EventBus(),
        Settings(_env_file=env_file),
        env_file=env_file,
        list_input_devices=lambda: INPUTS,
        capture_status=lambda: status,
    )
    with TestClient(app) as client:
        body = client.get("/audio-devices").json()
        assert body["error"] == "[Errno -9999] Unanticipated host error"
        assert body["capturing"] is None

        assert client.get("/mode").json()["audio_error"] == "[Errno -9999] Unanticipated host error"

        status.update(capturing="Realtek Mic", error=None)
        body = client.get("/audio-devices").json()
        assert (body["capturing"], body["error"]) == ("Realtek Mic", None)
        mode = client.get("/mode").json()
        assert (mode["audio_capturing"], mode["audio_error"]) == ("Realtek Mic", None)
