"""The overlay page rendered in headless Chromium, fed by the real server.

Needs Playwright and its browser: ``uv run playwright install chromium``.
Skips when either is missing.
"""

from __future__ import annotations

import json
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
    # A bounded graceful shutdown still runs the app's lifespan shutdown, so slices stop.
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", timeout_graceful_shutdown=3
    )
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
    thread.join(timeout=15)
    assert not thread.is_alive(), "overlay server did not stop"


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
    # A context per caption, so closing it also closes its /ws and keep-alive sockets.
    context = browser.new_context(viewport={"width": width, "height": width * 9 // 16})
    page = context.new_page()
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
        # The overlay shows the stage when a subtitle arrives, after rendering its rows.
        page.wait_for_selector("#stage:not(.hidden) #lines .line", state="attached")
        return page.evaluate(_BOXES_JS)
    finally:
        context.close()


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


KOREAN = (
    "하나님은 그렇지 않다는 것을 알고 계셨고, 그래서 하나님은 기본적으로, 아시다시피, "
    "우리 Z세대와 알파 세대에게, 그리고 그들의 삶에 고통이 있을 것이라고 말씀하셨습니다"
)
ENGLISH = (
    "And so when we read the opening chapters of this book we see that real suffering "
    "came suddenly and without any warning at all into the life of Job and his family"
)


@pytest.mark.parametrize("width", [1280, 1920])
@pytest.mark.parametrize("text", [KOREAN, ENGLISH], ids=["korean", "english"])
def test_each_caption_box_shows_one_row(browser, server, width, text) -> None:
    boxes = render_caption(browser, server, width, text, line_count=2)
    assert boxes
    assert [box["rows"] for box in boxes] == [1] * len(boxes)
    assert sum(box["rows"] for box in boxes) <= 2


@pytest.mark.parametrize("width", [1280, 1920])
def test_korean_caption_breaks_only_at_spaces(browser, server, width) -> None:
    boxes = render_caption(browser, server, width, KOREAN, line_count=2)
    shown = [word for box in boxes for word in box["text"].split()]
    words = KOREAN.split()
    assert shown == words[len(words) - len(shown) :]


def test_a_caption_stays_up_through_a_long_pause_until_captions_stop(browser, server) -> None:
    context = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = context.new_page()
    sockets: list[object] = []
    # The test plays the server's /ws, so it controls what the overlay hears and when.
    page.route_web_socket("**/ws", lambda ws: sockets.append(ws))
    page.clock.install()
    try:
        page.goto(f"{server}/overlay")
        deadline = time.monotonic() + 10
        while not sockets:
            if time.monotonic() > deadline:
                raise RuntimeError("overlay did not open /ws")
            page.wait_for_timeout(50)
        line = {"type": "subtitle", "text": "A long sentence is coming.", "provisional": False}
        sockets[0].send(json.dumps(line))
        page.wait_for_selector("#stage:not(.hidden) #lines .line", state="attached")
        # The speaker is mid-sentence for a minute; the last line must not vanish meanwhile.
        page.clock.run_for(60_000)
        assert page.locator("#stage.hidden").count() == 0
        assert page.locator("#lines .line").all_text_contents() == ["A long sentence is coming."]
        sockets[0].send(json.dumps({"type": "captions_state", "is_active": False}))
        page.wait_for_selector("#stage.hidden", state="attached")
        assert page.locator("#lines .line").count() == 0
    finally:
        context.close()


# A 100-character en_to_ko line from a real run (2026-11-10): three rows at the default font.
LONG_KOREAN = (
    "왜냐하면 이건 큰 주제거든요. 저는 이 모든 설명을 다 할 수는 없겠지만, 기독교 신앙이 "
    "이 문제를 어떻게 다루는지에 대해, 저는 지치신 분들이나 이 문제로 고군분투하는 분들에게,"
)


def test_a_long_korean_line_shows_whole_at_the_default_line_count(browser, server) -> None:
    # Omitting line_count takes the default, as a booth that never changed it does.
    style = {k: v for k, v in DEFAULT_STYLE.items() if k != "line_count"}
    httpx.put(f"{server}/overlay-style", json=style).raise_for_status()
    context = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = context.new_page()
    try:
        clients = httpx.get(f"{server}/health").json()["clients"]
        page.goto(f"{server}/overlay")
        deadline = time.monotonic() + 10
        while httpx.get(f"{server}/health").json()["clients"] <= clients:
            if time.monotonic() > deadline:
                raise RuntimeError("overlay did not open /ws")
            time.sleep(0.05)
        httpx.post(f"{server}/inject", json={"text": LONG_KOREAN}).raise_for_status()
        page.wait_for_selector("#stage:not(.hidden) #lines .line", state="attached")
        boxes = page.evaluate(_BOXES_JS)
    finally:
        context.close()
    assert len(boxes) == 3
    assert " ".join(box["text"] for box in boxes).split() == LONG_KOREAN.split()
