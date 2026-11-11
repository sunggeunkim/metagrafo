from fastapi.testclient import TestClient

from main import create_app


def test_quit_asks_the_server_to_stop() -> None:
    asked: list[bool] = []
    with TestClient(create_app(request_quit=lambda: asked.append(True))) as client:
        response = client.post("/quit")
        assert response.status_code == 200
        assert response.json() == {"quitting": True}
    assert asked == [True]


def test_quit_is_refused_when_nothing_can_stop_the_server() -> None:
    # `uvicorn main:app` and tests run the module app, which has no server to stop.
    with TestClient(create_app()) as client:
        response = client.post("/quit")
        assert response.status_code == 409
        assert "console" in response.json()["detail"]
