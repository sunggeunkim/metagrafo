"""The overlay page rendered in headless Chromium, fed by the real server.

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

from main import create_app

sync_api = pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")

DEFAULT_STYLE = {
    "position": "top_left",
    "font_size_vw": 2.5,
    "inset_vertical_pct": 1.0,
    "inset_horizontal_pct": 1.0,
    "box_width_pct": 100.0,
    "box_height_pct": 30.0,
    "line_count": 2,
}

# Measured per shown box: its text's rendered line count and its rectangle.
_BOXES_JS = """
() => [...document.querySelectorAll('#lines .line')].map((el) => {
  const range = document.createRange();
  range.selectNodeContents(el);
  const tops = [];
  for (const rect of range.getClientRects()) {
    if (rect.width > 0 && !tops.some((t) => Math.abs(t - rect.top) < 2)) {
      tops.push(rect.top);
    }
  }
  const box = el.getBoundingClientRect();
  return {
    text: el.textContent,
    rows: tops.length,
    left: box.left,
    right: box.right,
    top: box.top,
    bottom: box.bottom,
    fontSize: parseFloat(getComputedStyle(el).fontSize),
  };
})
"""


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    app = create_app(
        env_file=tmp_path_factory.mktemp("env") / ".env",
        list_input_devices=lambda: [],
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    uv_server = uvicorn.Server(config)
    thread = threading.Thread(target=uv_server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not uv_server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("overlay server did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    uv_server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def browser() -> Iterator[object]:
    with sync_api.sync_playwright() as playwright:
        try:
            chromium = playwright.chromium.launch()
        except sync_api.Error as exc:
            pytest.skip(f"Chromium for Playwright is not installed: {exc}")
        yield chromium
        chromium.close()


def render_caption(
    browser: object, server: str, width: int, text: str, **style: object
) -> list[dict]:
    """Open /overlay at width x 16:9, set the style, inject one caption, return its boxes."""
    page = browser.new_page(viewport={"width": width, "height": width * 9 // 16})
    try:
        clients = httpx.get(f"{server}/health").json()["clients"]
        page.goto(f"{server}/overlay")
        deadline = time.monotonic() + 10
        while httpx.get(f"{server}/health").json()["clients"] <= clients:
            if time.monotonic() > deadline:
                raise RuntimeError("overlay did not open /ws")
            time.sleep(0.05)
        httpx.put(f"{server}/overlay-style", json={**DEFAULT_STYLE, **style}).raise_for_status()
        httpx.post(f"{server}/inject", json={"text": text}).raise_for_status()
        last_word = text.split()[-1]
        page.wait_for_function(
            "(word) => [...document.querySelectorAll('#lines .line')]"
            ".some((el) => el.textContent.endsWith(word))",
            arg=last_word,
        )
        return page.evaluate(_BOXES_JS)
    finally:
        page.close()


@pytest.mark.parametrize("width", [1280, 1920])
@pytest.mark.parametrize("position", ["top_left", "top_center", "bottom_center"])
def test_short_caption_renders_in_each_position(browser, server, width, position) -> None:
    height = width * 9 // 16
    boxes = render_caption(browser, server, width, "Hello, everyone.", position=position)
    assert [box["text"] for box in boxes] == ["Hello, everyone."]
    box = boxes[0]
    assert box["rows"] == 1
    assert box["fontSize"] == pytest.approx(width * 0.025, abs=0.5)
    middle = (box["left"] + box["right"]) / 2
    if position == "top_left":
        assert box["left"] == pytest.approx(width * 0.01, abs=2)
        assert box["top"] < height * 0.05
    elif position == "top_center":
        assert middle == pytest.approx(width / 2, abs=2)
        assert box["top"] < height * 0.05
    else:
        assert middle == pytest.approx(width / 2, abs=2)
        assert box["bottom"] > height * 0.95


def test_font_size_and_box_width_settings_apply(browser, server) -> None:
    long_text = " ".join(["Hello, everyone."] * 6)
    boxes = render_caption(
        browser, server, 1920, long_text, font_size_vw=4.0, box_width_pct=50.0
    )
    assert boxes
    for box in boxes:
        assert box["fontSize"] == pytest.approx(1920 * 0.04, abs=0.5)
        assert box["right"] - box["left"] <= 1920 * 0.5 + 1
