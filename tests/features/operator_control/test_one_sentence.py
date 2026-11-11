import pytest
from fastapi.testclient import TestClient

from core.settings import Settings
from main import create_app


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ONE_SENTENCE", raising=False)
    path = tmp_path / ".env"
    path.write_text("AUDIO_DEVICE_NAME=Booth Mic\n", encoding="utf-8")
    return path


def _app(env_file):
    settings = Settings(caption_engine="whisper", gemini_api_key="", _env_file=env_file)
    return create_app(settings=settings, env_file=env_file)


def _hello_style(client: TestClient) -> dict:
    with client.websocket_connect("/ws") as ws:
        style = ws.receive_json()
        ws.receive_json()  # captions_state
    assert style["type"] == "overlay_style"
    return style


def test_one_sentence_at_a_time_is_on_by_default_and_the_overlay_is_told(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        assert client.get("/mode").json()["one_sentence"] is True
        assert _hello_style(client)["one_sentence"] is True


def test_turning_it_off_tells_the_open_overlay_and_is_kept_for_next_time(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.receive_json()
            response = client.put("/one-sentence", json={"one_sentence": False})
            assert response.status_code == 200
            assert response.json()["one_sentence"] is False
            message = ws.receive_json()
        assert message["type"] == "overlay_style"
        assert message["one_sentence"] is False
    assert "ONE_SENTENCE=false" in env_file.read_text(encoding="utf-8")
    with TestClient(_app(env_file)) as client:
        assert client.get("/mode").json()["one_sentence"] is False
        assert _hello_style(client)["one_sentence"] is False


def test_a_style_change_keeps_the_one_sentence_choice(env_file) -> None:
    with TestClient(_app(env_file)) as client:
        client.put("/one-sentence", json={"one_sentence": False})
        style = client.get("/mode").json()
        body = {
            "position": style["position"],
            "font_size_vw": 3.0,
            "inset_vertical_pct": style["inset_vertical_pct"],
            "inset_horizontal_pct": style["inset_horizontal_pct"],
            "box_width_pct": style["box_width_pct"],
            "box_height_pct": style["box_height_pct"],
            "line_count": style["line_count"],
        }
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.receive_json()
            client.put("/overlay-style", json=body)
            message = ws.receive_json()
    assert message["font_size_vw"] == 3.0
    assert message["one_sentence"] is False
