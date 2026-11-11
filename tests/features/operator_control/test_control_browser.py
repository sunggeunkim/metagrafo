"""Booth control rendered in headless Chromium, fed by the real app.

Needs Playwright and its browser: ``uv run playwright install chromium``.
Skips when either is missing.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator

import httpx
import pytest
import uvicorn

from core.settings import Settings
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


@pytest.fixture
def control_saving(tmp_path, monkeypatch) -> Iterator[tuple[str, object]]:
    """A served app that saves transcripts under tmp_path."""
    monkeypatch.delenv("SAVE_TRANSCRIPT", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    app = create_app(
        settings=Settings(caption_engine="whisper", gemini_api_key="", _env_file=env_file),
        env_file=env_file,
        list_input_devices=lambda: [],
        transcripts_dir=tmp_path / "transcripts",
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
    yield f"http://127.0.0.1:{port}", tmp_path / "transcripts"
    server.should_exit = True
    thread.join(timeout=15)


def test_save_transcript_checkbox_saves_to_a_file_it_names(browser, control_saving) -> None:
    url, folder = control_saving
    page = browser.new_page()
    try:
        page.goto(f"{url}/control")
        page.wait_for_selector("#transcript-status:has-text('Not saving')")
        assert not page.is_checked("#save-transcript")

        page.check("#save-transcript")
        page.wait_for_selector("#transcript-status:has-text('Saving to')")
        [saved] = list(folder.glob("*.jsonl"))
        assert saved.name in page.inner_text("#transcript-status")

        page.uncheck("#save-transcript")
        page.wait_for_selector("#transcript-status:has-text('Not saving')")
    finally:
        page.close()


def test_additional_context_box_saves_and_comes_back_on_reload(browser, control_saving) -> None:
    url, _folder = control_saving
    page = browser.new_page()
    try:
        page.goto(f"{url}/control")
        page.wait_for_selector("#cleanup-context-status:has-text('No additional context')")
        assert page.locator("h2", has_text="Additional context").count() == 1
        # Nothing saved means an empty box, with no example text shown in it either.
        assert page.input_value("#cleanup-context") == ""
        assert page.get_attribute("#cleanup-context", "placeholder") in (None, "")
        page.fill("#cleanup-context", "Sermon: the problem of evil.\nScripture: Job 1-2.")
        page.click("#cleanup-context-save")
        page.wait_for_selector("#cleanup-context-status:has-text('Saved')")

        page.reload()
        page.wait_for_function(
            "document.getElementById('cleanup-context').value.includes('Job 1-2')"
        )
        assert page.input_value("#cleanup-context") == "Sermon: the problem of evil.\nScripture: Job 1-2."

        page.click("#cleanup-context-clear")
        page.wait_for_selector("#cleanup-context-status:has-text('No additional context')")
        assert page.input_value("#cleanup-context") == ""
    finally:
        page.close()


def test_korean_style_box_starts_at_the_default_saves_turns_off_and_resets(browser, control_saving) -> None:
    from core.settings import KOREAN_STYLE_DEFAULT

    url, _folder = control_saving
    youth = "청소년 예배입니다. 해요체로 친근하게 쓰십시오."
    page = browser.new_page()
    try:
        page.goto(f"{url}/control")
        page.wait_for_selector("#korean-style-status:has-text('Default')")
        assert page.input_value("#korean-style") == KOREAN_STYLE_DEFAULT

        page.fill("#korean-style", youth)
        page.click("#korean-style-save")
        page.wait_for_selector("#korean-style-status:has-text('Saved')")
        page.reload()
        page.wait_for_function("document.getElementById('korean-style').value.includes('청소년')")
        assert page.input_value("#korean-style") == youth

        page.fill("#korean-style", "")
        page.click("#korean-style-save")
        page.wait_for_selector("#korean-style-status:has-text('Off')")

        page.click("#korean-style-reset")
        page.wait_for_selector("#korean-style-status:has-text('Default')")
        assert page.input_value("#korean-style") == KOREAN_STYLE_DEFAULT
    finally:
        page.close()


def test_one_sentence_checkbox_starts_on_and_lines_on_screen_waits_for_it_off(browser, control_saving) -> None:
    url, _folder = control_saving
    page = browser.new_page()
    try:
        page.goto(f"{url}/control")
        page.wait_for_function("document.getElementById('one-sentence').checked")
        assert page.is_disabled("#line-count")

        page.uncheck("#one-sentence")
        page.wait_for_function("!document.getElementById('line-count').disabled")
        assert httpx.get(f"{url}/mode").json()["one_sentence"] is False

        page.reload()
        page.wait_for_function("!document.getElementById('line-count').disabled")
        assert not page.is_checked("#one-sentence")
    finally:
        page.close()
