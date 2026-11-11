"""Booth control rendered in headless Chromium, fed by the real app.

Needs Playwright and its browser: ``uv run playwright install chromium``.
Skips when either is missing.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn

from main import create_app

sync_api = pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")


@pytest.fixture
def control(tmp_path) -> Iterator[tuple[str, list[bool]]]:
    """A served app whose quit is only recorded, so the server stays up for the page."""
    asked: list[bool] = []
    app = create_app(
        env_file=tmp_path / ".env",
        list_input_devices=lambda: [],
        request_quit=lambda: asked.append(True),
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", timeout_graceful_shutdown=3
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("control server did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", asked
    server.should_exit = True
    thread.join(timeout=15)


@pytest.fixture(scope="module")
def browser() -> Iterator[object]:
    with sync_api.sync_playwright() as playwright:
        try:
            chromium = playwright.chromium.launch()
        except sync_api.Error as exc:
            pytest.skip(f"Chromium for Playwright is not installed: {exc}")
        yield chromium
        chromium.close()


def test_quit_button_stops_metagrafo_once_confirmed(browser, control) -> None:
    url, asked = control
    page = browser.new_page()
    try:
        page.goto(f"{url}/control")
        page.once("dialog", lambda dialog: dialog.accept())
        page.click("#quit")
        page.wait_for_selector("text=Metagrafo has stopped")
        assert asked == [True]
        assert "Start Metagrafo" in page.inner_text("body")
    finally:
        page.close()


def test_quit_button_does_nothing_when_cancelled(browser, control) -> None:
    url, asked = control
    page = browser.new_page()
    try:
        page.goto(f"{url}/control")
        page.once("dialog", lambda dialog: dialog.dismiss())
        page.click("#quit")
        page.wait_for_timeout(500)
        assert asked == []
        assert page.is_visible("#captions-toggle")
    finally:
        page.close()
