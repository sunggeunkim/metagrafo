from fastapi.testclient import TestClient

from main import create_app


def test_overlay_is_two_line_browser_source() -> None:
    with TestClient(create_app()) as client:
        page = client.get("/overlay")
        assert page.status_code == 200
        html = page.text
        assert "background: transparent" in html
        assert 'id="previous"' in html
        assert 'id="current"' in html
        assert "karaoke" not in html.lower()


def test_inject_broadcasts_subtitle_json_on_websocket() -> None:
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws") as ws:
            response = client.post("/inject", json={"text": "Hello, everyone."})
            assert response.status_code == 200
            payload = ws.receive_json()
    assert payload["type"] == "subtitle"
    assert payload["text"] == "Hello, everyone."
    assert payload["mode"] == "ko_to_en"
    assert payload["source_language"] == "ko"
    assert payload["target_language"] == "en"
    assert "chunk_id" in payload


def test_captions_off_broadcasts_clear_on_websocket() -> None:
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws") as ws:
            client.put("/captions", json={"is_active": False})
            payload = ws.receive_json()
    assert payload == {"type": "captions_state", "is_active": False}


def test_mode_change_broadcasts_on_websocket() -> None:
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws") as ws:
            client.put("/mode", json={"mode": "en_to_en"})
            payload = ws.receive_json()
    assert payload["type"] == "mode"
    assert payload["mode"] == "en_to_en"
    assert payload["source_language"] == "en"
    assert payload["target_language"] == "en"
