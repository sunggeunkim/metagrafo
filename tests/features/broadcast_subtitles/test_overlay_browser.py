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
    browser: object, server: str, width: int, text: str, *, one_sentence: bool = False, **style: object
) -> list[dict]:
    """Open /overlay at width x 16:9, set the style, inject one caption, return its boxes.

    Rolling rows by default, since these tests are about how one line is cut into rows.
    """
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
        httpx.put(f"{server}/one-sentence", json={"one_sentence": one_sentence}).raise_for_status()
        httpx.put(f"{server}/overlay-style", json={**DEFAULT_STYLE, **style}).raise_for_status()
        httpx.post(f"{server}/inject", json={"text": text}).raise_for_status()
        # The overlay shows the stage when a subtitle arrives, after rendering its rows.
        page.wait_for_selector("#stage:not(.hidden) #lines .line", state="attached")
        return page.evaluate(_BOXES_JS)
    finally:
        context.close()


@pytest.mark.parametrize("width", [1280, 1920])
@pytest.mark.parametrize("position", ["top_left", "top_center", "bottom_left", "bottom_center"])
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
    elif position == "bottom_left":
        assert box["left"] == pytest.approx(width * 0.01, abs=2)
        assert box["bottom"] > height * 0.95
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


# Seen in OBS 33 (Chromium 150), 2026-11-11: this line showed as "… 아닐 것" (wrapping "것" onto a
# second row inside its box) and then "입니다." on its own. Newer Chromium also reports a zero-width
# rect at the end of the row above for the space where a row wraps; this script makes the first
# character after that space report it too, the way the measuring in OBS behaved.
WRAP_BOUNDARY_RECTS = """
(() => {
  const own = Range.prototype.getClientRects;
  Range.prototype.getClientRects = function () {
    const rects = [...own.call(this)];
    const node = this.startContainer;
    if (node.nodeType === Node.TEXT_NODE && this.startOffset > 0 && node.data[this.startOffset - 1] === " ") {
      const space = document.createRange();
      space.setStart(node, this.startOffset - 1);
      space.setEnd(node, this.startOffset);
      const before = own.call(space)[0];
      if (before) rects.unshift(new DOMRect(before.right, before.top, 0, before.height));
    }
    return rects;
  };
  Range.prototype.getBoundingClientRect = function () {
    const rects = this.getClientRects();
    if (!rects.length) return new DOMRect();
    const left = Math.min(...rects.map((r) => r.left));
    const top = Math.min(...rects.map((r) => r.top));
    const right = Math.max(...rects.map((r) => r.right));
    const bottom = Math.max(...rects.map((r) => r.bottom));
    return new DOMRect(left, top, right - left, bottom - top);
  };
})();
"""

OBS_SPLIT_LINE = "그래서 이번 설교의 많은 부분이 여러분의 머리를 위한 것이고, 마음을 위한 것은 아닐 것입니다."


def test_a_row_starts_at_its_first_word_when_the_wrap_also_reports_a_rect_on_the_row_above(
    browser, server
) -> None:
    httpx.put(f"{server}/overlay-style", json={**DEFAULT_STYLE, "line_count": 3}).raise_for_status()
    context = browser.new_context(viewport={"width": 1920, "height": 1080})
    context.add_init_script(WRAP_BOUNDARY_RECTS)
    page = context.new_page()
    try:
        clients = httpx.get(f"{server}/health").json()["clients"]
        page.goto(f"{server}/overlay")
        deadline = time.monotonic() + 10
        while httpx.get(f"{server}/health").json()["clients"] <= clients:
            if time.monotonic() > deadline:
                raise RuntimeError("overlay did not open /ws")
            time.sleep(0.05)
        httpx.post(f"{server}/inject", json={"text": OBS_SPLIT_LINE}).raise_for_status()
        page.wait_for_selector("#stage:not(.hidden) #lines .line", state="attached")
        boxes = page.evaluate(_BOXES_JS)
    finally:
        context.close()
    assert [box["text"] for box in boxes] == [
        "그래서 이번 설교의 많은 부분이 여러분의 머리를 위한 것이고, 마음을 위한 것은 아닐",
        "것입니다.",
    ]
    assert [box["rows"] for box in boxes] == [1, 1]


def test_one_sentence_at_a_time_shows_the_newest_sentence_whole_then_replaces_it(browser, server) -> None:
    httpx.put(f"{server}/one-sentence", json={"one_sentence": True}).raise_for_status()
    # Fewer rows than the sentence needs: in this mode the setting must not cut it.
    httpx.put(f"{server}/overlay-style", json={**DEFAULT_STYLE, "line_count": 2}).raise_for_status()
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
        first = page.evaluate(_BOXES_JS)
        httpx.post(f"{server}/inject", json={"text": "Then the next sentence."}).raise_for_status()
        page.wait_for_function(
            "document.getElementById('lines').textContent.includes('Then the next sentence.')"
        )
        second = page.evaluate(_BOXES_JS)
    finally:
        context.close()
    assert len(first) == 3
    assert [box["rows"] for box in first] == [1, 1, 1]
    assert " ".join(box["text"] for box in first).split() == LONG_KOREAN.split()
    assert [box["text"] for box in second] == ["Then the next sentence."]


def test_one_sentence_taller_than_the_box_keeps_its_end_and_drops_its_start(browser, server) -> None:
    httpx.put(f"{server}/one-sentence", json={"one_sentence": True}).raise_for_status()
    # A box about two rows tall, for a sentence that needs three.
    style = {**DEFAULT_STYLE, "line_count": 3, "box_height_pct": 15.0}
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
        stage = page.evaluate(
            "(() => { const r = document.getElementById('stage').getBoundingClientRect();"
            " return {top: r.top, bottom: r.bottom}; })()"
        )
    finally:
        context.close()
    words = LONG_KOREAN.split()
    shown = " ".join(box["text"] for box in boxes).split()
    assert 0 < len(boxes) < 3
    # The end of the sentence is what is shown; its start is what leaves.
    assert shown == words[len(words) - len(shown) :]
    for box in boxes:
        assert box["top"] >= stage["top"] - 1
        assert box["bottom"] <= stage["bottom"] + 1


TWO_ROW_KOREAN = (
    "왜냐하면 이건 큰 주제거든요. 저는 이 모든 설명을 다 할 수는 없겠지만, 기독교 신앙이 "
    "이 문제를 어떻게 다루는지에 대해 말씀드리겠습니다."
)


def _one_sentence_screens(browser: object, server: str, texts: list[str]) -> list[list[str]]:
    """Inject finished sentences one by one in one-sentence mode; return the rows after each."""
    httpx.put(f"{server}/one-sentence", json={"one_sentence": True}).raise_for_status()
    httpx.put(f"{server}/overlay-style", json=DEFAULT_STYLE).raise_for_status()
    context = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = context.new_page()
    screens = []
    try:
        clients = httpx.get(f"{server}/health").json()["clients"]
        page.goto(f"{server}/overlay")
        deadline = time.monotonic() + 10
        while httpx.get(f"{server}/health").json()["clients"] <= clients:
            if time.monotonic() > deadline:
                raise RuntimeError("overlay did not open /ws")
            time.sleep(0.05)
        for text in texts:
            httpx.post(f"{server}/inject", json={"text": text}).raise_for_status()
            last_word = text.split()[-1]
            page.wait_for_function(
                f"[...document.querySelectorAll('#lines .line')].some((el) => el.textContent.endsWith({last_word!r}))"
            )
            screens.append([box["text"] for box in page.evaluate(_BOXES_JS)])
    finally:
        context.close()
    return screens


def test_two_lines_at_a_time_keeps_a_one_line_sentence_under_the_one_before(browser, server) -> None:
    screens = _one_sentence_screens(
        browser, server, ["First short sentence.", "Second short sentence.", "Third short sentence."]
    )
    assert screens == [
        ["First short sentence."],
        ["First short sentence.", "Second short sentence."],
        # The screen is full at two lines, so the next sentence starts afresh.
        ["Third short sentence."],
    ]


def test_two_lines_at_a_time_never_shows_more_than_two_lines(browser, server) -> None:
    screens = _one_sentence_screens(browser, server, ["First short sentence.", TWO_ROW_KOREAN])
    assert screens[0] == ["First short sentence."]
    # One line plus two would be three, so the two-line sentence starts afresh.
    assert len(screens[1]) == 2
    assert " ".join(screens[1]).split() == TWO_ROW_KOREAN.split()
