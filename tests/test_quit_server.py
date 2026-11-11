"""The server main() runs stops when booth control asks it to quit."""

from __future__ import annotations

import socket
import threading
import time

import httpx

from main import build_server, create_app


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_quit_stops_the_server_after_answering() -> None:
    port = _free_port()
    server = build_server(lambda request_quit: create_app(request_quit=request_quit), port=port)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)

    response = httpx.post(f"http://127.0.0.1:{port}/quit", timeout=5)

    assert response.status_code == 200
    assert response.json() == {"quitting": True}
    thread.join(timeout=15)
    assert not thread.is_alive(), "server kept running after /quit"
