from fastapi.testclient import TestClient

from main import create_app


def _skip_hello(ws):
    hello = ws.receive_json()
    assert hello["type"] == "overlay_style"
    return hello


def test_overlay_is_two_line_browser_source() -> None:
    with TestClient(create_app()) as client:
        page = client.get("/overlay")
        assert page.status_code == 200
        html = page.text
        assert "background: transparent" in html
        assert 'id="previous"' in html
        assert 'id="current"' in html
        assert "karaoke" not in html.lower()
        assert "--box-w" in html
        assert "--box-h" in html
        assert "max-height: var(--box-h)" in html
        assert "width: 70%" not in html
        assert "no-store" in page.headers.get("cache-control", "")


def test_inject_broadcasts_subtitle_json_on_websocket() -> None:
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws") as ws:
            _skip_hello(ws)
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
            _skip_hello(ws)
            client.put("/captions", json={"is_active": False})
            payload = ws.receive_json()
    assert payload == {"type": "captions_state", "is_active": False}


def test_mode_change_broadcasts_on_websocket() -> None:
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws") as ws:
            _skip_hello(ws)
            client.put("/mode", json={"mode": "en_to_en"})
            payload = ws.receive_json()
    assert payload["type"] == "mode"
    assert payload["mode"] == "en_to_en"
    assert payload["source_language"] == "en"
    assert payload["target_language"] == "en"


def test_overlay_style_put_broadcasts_on_websocket() -> None:
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws") as ws:
            hello = _skip_hello(ws)
            assert hello["position"] == "top_left"
            assert hello["font_size_vw"] == 2.5
            assert hello["inset_vertical_pct"] == 1.0
            assert hello["inset_horizontal_pct"] == 1.0
            assert hello["box_width_pct"] == 100.0
            assert hello["box_height_pct"] == 30.0
            response = client.put(
                "/overlay-style",
                json={
                    "position": "bottom_center",
                    "font_size_vw": 5.0,
                    "inset_vertical_pct": 0.0,
                    "inset_horizontal_pct": 1.5,
                    "box_width_pct": 100.0,
                    "box_height_pct": 20.0,
                },
            )
            assert response.status_code == 200
            payload = ws.receive_json()
    assert payload == {
        "type": "overlay_style",
        "position": "bottom_center",
        "font_size_vw": 5.0,
        "inset_vertical_pct": 0.0,
        "inset_horizontal_pct": 1.5,
        "box_width_pct": 100.0,
        "box_height_pct": 20.0,
    }


def test_overlay_style_rejects_out_of_range_font() -> None:
    with TestClient(create_app()) as client:
        response = client.put(
            "/overlay-style",
            json={
                "position": "top_left",
                "font_size_vw": 20,
                "inset_vertical_pct": 8.0,
                "inset_horizontal_pct": 5.0,
                "box_width_pct": 100.0,
                "box_height_pct": 10.0,
            },
        )
        assert response.status_code == 422


def test_overlay_style_rejects_out_of_range_inset() -> None:
    with TestClient(create_app()) as client:
        too_high = client.put(
            "/overlay-style",
            json={
                "position": "top_left",
                "font_size_vw": 3.2,
                "inset_vertical_pct": 21,
                "inset_horizontal_pct": 5.0,
                "box_width_pct": 100.0,
                "box_height_pct": 10.0,
            },
        )
        assert too_high.status_code == 422
        too_low = client.put(
            "/overlay-style",
            json={
                "position": "top_left",
                "font_size_vw": 3.2,
                "inset_vertical_pct": 8.0,
                "inset_horizontal_pct": -1,
                "box_width_pct": 100.0,
                "box_height_pct": 10.0,
            },
        )
        assert too_low.status_code == 422


def test_overlay_style_rejects_out_of_range_box() -> None:
    with TestClient(create_app()) as client:
        too_narrow = client.put(
            "/overlay-style",
            json={
                "position": "top_left",
                "font_size_vw": 3.2,
                "inset_vertical_pct": 8.0,
                "inset_horizontal_pct": 5.0,
                "box_width_pct": 5.0,
                "box_height_pct": 10.0,
            },
        )
        assert too_narrow.status_code == 422
        too_short = client.put(
            "/overlay-style",
            json={
                "position": "top_left",
                "font_size_vw": 3.2,
                "inset_vertical_pct": 8.0,
                "inset_horizontal_pct": 5.0,
                "box_width_pct": 100.0,
                "box_height_pct": 1.0,
            },
        )
        assert too_short.status_code == 422


def test_overlay_html_has_position_classes() -> None:
    with TestClient(create_app()) as client:
        html = client.get("/overlay").text
        assert "pos-top_left" in html
        assert "pos-bottom_center" in html
        assert "font_size_vw" in html
        assert "--inset-v" in html
        assert "--inset-h" in html
        assert "var(--inset-v)" in html
        assert "var(--inset-h)" in html
        assert "var(--box-w)" in html
        assert "var(--box-h)" in html
        assert "overflow-wrap: normal" in html
        assert "function fitCaptionBox" not in html
