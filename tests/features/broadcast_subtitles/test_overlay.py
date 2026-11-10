import asyncio

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketState

from core.event_bus import EventBus
from features.broadcast_subtitles.hub import SubtitleHub
from features.operator_control.events import CaptionsStateEvent
from features.translate_speech.events import SubtitleEvent
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
        assert "background: rgba(0, 0, 0, 0.55)" in html
        assert 'id="lines"' in html
        assert "line_count" in html
        assert 'id="previous"' not in html
        assert "karaoke" not in html.lower()
        assert "--box-w" in html
        assert "--box-h" in html
        assert "max-height: var(--box-h)" in html
        assert "width: 70%" not in html
        assert "no-store" in page.headers.get("cache-control", "")


class _Socket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.client_state = WebSocketState.CONNECTED

    async def accept(self) -> None:
        return None

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)


async def test_hub_marks_a_growing_line_provisional() -> None:
    bus = EventBus()
    hub = SubtitleHub(bus)
    socket = _Socket()
    await hub.connect(socket)
    await bus.publish(
        SubtitleEvent(
            chunk_id="c1",
            text="Today's topical",
            source_language="ko",
            target_language="en",
            mode="ko_to_en",
            created_at=1.0,
            duration_s=0.0,
            provisional=True,
        )
    )
    assert socket.sent[-1]["type"] == "subtitle"
    assert socket.sent[-1]["text"] == "Today's topical"
    assert socket.sent[-1]["provisional"] is True


def test_overlay_grows_the_open_line_in_place() -> None:
    with TestClient(create_app()) as client:
        html = client.get("/overlay").text
    assert "showDraft" in html
    assert "draftText" in html
    assert "measureRows" in html
    assert "slice(-lineCount)" in html
    assert "msg.provisional" in html
    assert "max-height: var(--box-h)" in html
    assert "calc(var(--line-count)" not in html


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
            assert hello["line_count"] == 2
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
        "line_count": 2,
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


def _line(text: str, *, paced: bool = True) -> SubtitleEvent:
    return SubtitleEvent(
        chunk_id=text,
        text=text,
        source_language="ko",
        target_language="en",
        mode="ko_to_en",
        created_at=1.0,
        duration_s=0.0,
        paced=paced,
    )


def _shown(socket: _Socket) -> list[str]:
    return [m["text"] for m in socket.sent if m["type"] == "subtitle"]


async def test_a_paced_line_stays_up_its_minimum_time_before_the_next() -> None:
    bus = EventBus()
    hub = SubtitleHub(bus, min_on_screen_s=0.3, chars_per_s=1000.0)
    socket = _Socket()
    await hub.connect(socket)
    await bus.publish(_line("First line."))
    await bus.publish(_line("Second line."))
    await asyncio.sleep(0.2)
    assert _shown(socket) == ["First line."]
    await asyncio.sleep(0.2)
    assert _shown(socket) == ["First line.", "Second line."]


async def test_a_long_paced_line_stays_up_long_enough_to_read_its_characters() -> None:
    bus = EventBus()
    # 40 characters at 100 per second: 0.4 s, longer than the 0.05 s minimum.
    hub = SubtitleHub(bus, min_on_screen_s=0.05, chars_per_s=100.0)
    socket = _Socket()
    await hub.connect(socket)
    await bus.publish(_line("A line of exactly forty characters long."))
    await bus.publish(_line("Next."))
    await asyncio.sleep(0.25)
    assert _shown(socket) == ["A line of exactly forty characters long."]
    await asyncio.sleep(0.3)
    assert _shown(socket) == ["A line of exactly forty characters long.", "Next."]


async def test_paced_lines_that_arrive_early_wait_and_show_in_order() -> None:
    bus = EventBus()
    hub = SubtitleHub(bus, min_on_screen_s=0.2, chars_per_s=1000.0)
    socket = _Socket()
    await hub.connect(socket)
    await bus.publish(_line("One."))
    await bus.publish(_line("Two."))
    await bus.publish(_line("Three."))
    await asyncio.sleep(0.1)
    assert _shown(socket) == ["One."]
    await asyncio.sleep(0.2)
    assert _shown(socket) == ["One.", "Two."]
    await asyncio.sleep(0.2)
    assert _shown(socket) == ["One.", "Two.", "Three."]


async def test_an_unpaced_line_is_sent_at_once_while_paced_lines_wait() -> None:
    bus = EventBus()
    hub = SubtitleHub(bus, min_on_screen_s=0.3, chars_per_s=1000.0)
    socket = _Socket()
    await hub.connect(socket)
    await bus.publish(_line("Paced one."))
    await bus.publish(_line("Paced two."))
    await bus.publish(_line("Whisper line.", paced=False))
    assert _shown(socket) == ["Paced one.", "Whisper line."]
    await asyncio.sleep(0.4)
    assert _shown(socket) == ["Paced one.", "Whisper line.", "Paced two."]


async def test_a_backlog_too_far_behind_drops_the_oldest_waiting_line() -> None:
    bus = EventBus()
    hub = SubtitleHub(bus, min_on_screen_s=0.1, chars_per_s=1000.0, max_behind_s=0.25)
    socket = _Socket()
    await hub.connect(socket)
    await bus.publish(_line("A."))
    await bus.publish(_line("B."))
    await bus.publish(_line("C."))
    # D would show 0.3 s from now, past 0.25 s behind, so B (the oldest waiting) goes.
    await bus.publish(_line("D."))
    await asyncio.sleep(0.45)
    assert _shown(socket) == ["A.", "C.", "D."]


async def test_captions_off_clears_the_waiting_paced_lines() -> None:
    bus = EventBus()
    hub = SubtitleHub(bus, min_on_screen_s=0.1, chars_per_s=1000.0)
    socket = _Socket()
    await hub.connect(socket)
    await bus.publish(_line("Shown."))
    await bus.publish(_line("Waiting."))
    await bus.publish(CaptionsStateEvent(is_active=False))
    await asyncio.sleep(0.2)
    assert socket.sent[1:] == [
        {
            "type": "subtitle",
            "text": "Shown.",
            "chunk_id": "Shown.",
            "source_language": "ko",
            "target_language": "en",
            "mode": "ko_to_en",
            "ts": 1.0,
            "duration_s": 0.0,
            "provisional": False,
        },
        {"type": "captions_state", "is_active": False},
    ]


async def test_after_captions_off_the_next_paced_line_shows_at_once() -> None:
    bus = EventBus()
    hub = SubtitleHub(bus, min_on_screen_s=0.3, chars_per_s=1000.0)
    socket = _Socket()
    await hub.connect(socket)
    await bus.publish(_line("Before the song."))
    await bus.publish(CaptionsStateEvent(is_active=False))
    await bus.publish(CaptionsStateEvent(is_active=True))
    await bus.publish(_line("After the song."))
    assert _shown(socket) == ["Before the song.", "After the song."]


async def test_a_wait_from_before_captions_off_does_not_send_a_later_line_early() -> None:
    bus = EventBus()
    hub = SubtitleHub(bus, min_on_screen_s=0.3, chars_per_s=1000.0)
    socket = _Socket()
    await hub.connect(socket)
    await bus.publish(_line("A."))
    await bus.publish(_line("B."))
    await asyncio.sleep(0.15)
    await bus.publish(CaptionsStateEvent(is_active=False))
    await bus.publish(CaptionsStateEvent(is_active=True))
    await bus.publish(_line("C."))
    await bus.publish(_line("D."))
    # The wait for B would have ended at 0.3 s; D waits for C, until 0.45 s.
    await asyncio.sleep(0.23)
    assert _shown(socket) == ["A.", "C."]
    await asyncio.sleep(0.17)
    assert _shown(socket) == ["A.", "C.", "D."]


class _SlowSocket(_Socket):
    """Takes a while to send one line, so captions can go off part-way through a broadcast."""

    def __init__(self, slow_text: str) -> None:
        super().__init__()
        self._slow_text = slow_text

    async def send_json(self, data: dict) -> None:
        if data.get("text") == self._slow_text:
            await asyncio.sleep(0.2)
        self.sent.append(data)


async def test_captions_off_while_a_waiting_line_is_being_sent_stops_it_reaching_any_client() -> None:
    bus = EventBus()
    hub = SubtitleHub(bus, min_on_screen_s=0.1, chars_per_s=1000.0)
    sockets = [_SlowSocket("B."), _SlowSocket("B.")]
    for socket in sockets:
        await hub.connect(socket)
    await bus.publish(_line("A."))
    await bus.publish(_line("B."))
    # B starts going out at 0.1 s and the first send takes until 0.3 s.
    await asyncio.sleep(0.15)
    await bus.publish(CaptionsStateEvent(is_active=False))
    await asyncio.sleep(0.3)
    for socket in sockets:
        assert _shown(socket) == ["A."]
        assert socket.sent[-1] == {"type": "captions_state", "is_active": False}
