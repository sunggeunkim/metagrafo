from fastapi.testclient import TestClient

from core.event_bus import EventBus
from features.operator_control.events import CaptionsStateEvent, ModeChangedEvent
from main import create_app


def test_captions_default_on_and_sermon_mode() -> None:
    with TestClient(create_app()) as client:
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["ok"] is True
        assert body["captions_active"] is True
        assert body["mode"] == "ko_to_en"
        assert body["vad_min_silence_ms"] == 2000
        assert body["position"] == "top_left"
        assert body["font_size_vw"] == 2.5
        assert body["inset_vertical_pct"] == 1.0
        assert body["inset_horizontal_pct"] == 1.0
        assert body["box_width_pct"] == 100.0
        assert body["box_height_pct"] == 30.0
        assert "model" in body
        assert "device" in body
        assert "compute_type" in body


def test_turning_captions_on_publishes_state_event() -> None:
    bus = EventBus()
    seen: list[bool] = []

    async def handler(event: CaptionsStateEvent) -> None:
        seen.append(event.is_active)

    bus.subscribe(CaptionsStateEvent, handler)
    with TestClient(create_app(bus=bus)) as client:
        response = client.put("/captions", json={"is_active": True})
        assert response.status_code == 200
        assert response.json()["captions_active"] is True
        assert client.get("/health").json()["captions_active"] is True
    assert seen == [True]


def test_invalid_mode_is_rejected() -> None:
    with TestClient(create_app()) as client:
        response = client.put("/mode", json={"mode": "en_to_ko"})
        assert response.status_code == 422


def test_guest_mode_publishes_mode_changed() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler(event: ModeChangedEvent) -> None:
        seen.append(event.mode)

    bus.subscribe(ModeChangedEvent, handler)
    with TestClient(create_app(bus=bus)) as client:
        response = client.put("/mode", json={"mode": "en_to_en"})
        assert response.status_code == 200
        assert response.json()["mode"] == "en_to_en"
        assert client.get("/mode").json()["mode"] == "en_to_en"
    assert seen == ["en_to_en"]


def test_control_page_has_booth_buttons() -> None:
    with TestClient(create_app()) as client:
        page = client.get("/control")
        assert page.status_code == 200
        html = page.text
        assert "Captions" in html
        assert "ko_to_en" in html
        assert "en_to_en" in html
        assert "en_to_ko" not in html
        assert "VAD pause (ms)" in html
        assert "Font size (vw)" in html
        assert "Vertical inset (%)" in html
        assert "Horizontal inset (%)" in html
        assert "Box width (%)" in html
        assert "Box height (%)" in html
        assert "Reset Transform" in html


def test_vad_silence_ms_put_and_reject_out_of_range() -> None:
    with TestClient(create_app()) as client:
        ok = client.put("/vad-silence", json={"vad_min_silence_ms": 250})
        assert ok.status_code == 200
        assert ok.json()["vad_min_silence_ms"] == 250
        bad = client.put("/vad-silence", json={"vad_min_silence_ms": 50})
        assert bad.status_code == 422
