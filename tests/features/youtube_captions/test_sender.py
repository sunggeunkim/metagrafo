import asyncio
import logging
from datetime import UTC, datetime

import httpx
from fastapi.testclient import TestClient

from core.event_bus import EventBus
from core.settings import Settings
from features.translate_speech.events import SubtitleEvent
from features.youtube_captions.sender import (
    YouTubeCaptionSender,
    caption_body,
    format_timestamp,
    sequenced_url,
)
from features.youtube_captions.slice import YouTubeCaptionsSlice
from main import create_app

URL = "http://upload.youtube.com/closedcaption?cid=abc-123"
NOON = datetime(2026, 10, 25, 17, 0, 0, 250000, tzinfo=UTC).timestamp()


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class FakeYouTube:
    def __init__(self, statuses: list[int] | None = None) -> None:
        self.statuses = list(statuses or [])
        self.calls: list[dict] = []
        self.got = asyncio.Event()

    async def post(self, url, *, content, headers):
        self.calls.append({"url": url, "body": content.decode("utf-8"), "headers": headers})
        self.got.set()
        return _Response(self.statuses.pop(0) if self.statuses else 200)


def _line(
    text: str, *, provisional: bool = False, spoken_at: float | None = NOON
) -> SubtitleEvent:
    return SubtitleEvent(
        chunk_id="c1",
        text=text,
        source_language="ko",
        target_language="en",
        mode="ko_to_en",
        created_at=NOON + 3.0,
        duration_s=2.0,
        provisional=provisional,
        spoken_at=spoken_at,
    )


async def _run_until(fake: FakeYouTube, sender: YouTubeCaptionSender, count: int) -> None:
    task = asyncio.create_task(sender.run())
    try:
        async with asyncio.timeout(2):
            while len(fake.calls) < count:
                fake.got.clear()
                await fake.got.wait()
    finally:
        task.cancel()


def test_timestamp_is_utc_milliseconds_without_zone() -> None:
    assert format_timestamp(NOON) == "2026-10-25T17:00:00.250"


def test_body_is_timestamp_then_one_line_of_text() -> None:
    assert caption_body(NOON, "God is\ngood.  ") == "2026-10-25T17:00:00.250\nGod is good.\n"


def test_seq_joins_the_query_string() -> None:
    assert sequenced_url(URL, 3) == URL + "&seq=3"
    assert sequenced_url("http://h/closedcaption", 1) == "http://h/closedcaption?seq=1"


async def test_posts_finished_lines_in_order_with_rising_seq() -> None:
    bus = EventBus()
    fake = FakeYouTube()
    sender = YouTubeCaptionSender(bus, URL, post=fake.post)
    await bus.publish(_line("Hello, everyone."))
    await bus.publish(_line("Let us pray."))
    await _run_until(fake, sender, 2)

    assert [c["url"] for c in fake.calls] == [URL + "&seq=1", URL + "&seq=2"]
    assert fake.calls[0]["body"] == "2026-10-25T17:00:00.250\nHello, everyone.\n"
    assert fake.calls[0]["headers"]["Content-Type"] == "text/plain"
    assert sender.snapshot() == {"enabled": True, "sent": 2, "failed": 0, "last_status": 200}


async def test_line_without_a_speech_time_is_stamped_when_it_was_made() -> None:
    bus = EventBus()
    fake = FakeYouTube()
    sender = YouTubeCaptionSender(bus, URL, post=fake.post)
    await bus.publish(_line("Amen.", spoken_at=None))
    await _run_until(fake, sender, 1)

    assert fake.calls[0]["body"] == "2026-10-25T17:00:03.250\nAmen.\n"


async def test_skips_provisional_and_blank_lines() -> None:
    bus = EventBus()
    fake = FakeYouTube()
    sender = YouTubeCaptionSender(bus, URL, post=fake.post)
    await bus.publish(_line("Hello, every", provisional=True))
    await bus.publish(_line("   "))
    await bus.publish(_line("Hello, everyone."))
    await _run_until(fake, sender, 1)

    assert len(fake.calls) == 1
    assert fake.calls[0]["body"].endswith("\nHello, everyone.\n")


async def test_refused_line_keeps_seq_and_warns_once(caplog) -> None:
    bus = EventBus()
    fake = FakeYouTube([403, 403, 200])
    sender = YouTubeCaptionSender(bus, URL, post=fake.post)
    for text in ("one", "two", "three"):
        await bus.publish(_line(text))
    with caplog.at_level("DEBUG", logger="features.youtube_captions.sender"):
        await _run_until(fake, sender, 3)

    assert [c["url"][-5:] for c in fake.calls] == ["seq=1", "seq=1", "seq=1"]
    assert sender.sent == 1 and sender.failed == 2
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "abc-123" not in caplog.text


async def test_timeout_retries_the_same_line_and_seq() -> None:
    bus = EventBus()
    calls: list[tuple[str, bytes]] = []

    async def post(url, *, content, headers):
        calls.append((url, content))
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow")
        return _Response(200)

    sender = YouTubeCaptionSender(bus, URL, post=post, retry_delays_s=(0, 0))
    await bus.publish(_line("one"))
    await bus.publish(_line("two"))
    task = asyncio.create_task(sender.run())
    async with asyncio.timeout(2):
        while sender.sent < 2:
            await asyncio.sleep(0)
    task.cancel()
    assert [url[-5:] for url, _ in calls] == ["seq=1", "seq=1", "seq=2"]
    assert calls[0][1] == calls[1][1]
    assert sender.failed == 0


async def test_line_that_never_gets_through_is_dropped_and_seq_moves_on() -> None:
    bus = EventBus()
    calls: list[str] = []

    async def post(url, *, content, headers):
        calls.append(url)
        if b"one" in content:
            raise httpx.ConnectError("offline")
        return _Response(200)

    sender = YouTubeCaptionSender(bus, URL, post=post, retry_delays_s=(0, 0))
    await bus.publish(_line("one"))
    await bus.publish(_line("two"))
    task = asyncio.create_task(sender.run())
    async with asyncio.timeout(2):
        while sender.sent < 1:
            await asyncio.sleep(0)
    task.cancel()
    assert [url[-5:] for url in calls] == ["seq=1", "seq=1", "seq=1", "seq=2"]
    assert sender.failed == 1


async def test_real_httpx_requests_do_not_log_the_caption_key(caplog) -> None:
    bus = EventBus()
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200 if len(seen) > 1 else 403, text="2026-10-25T17:00:00.000")

    sender = YouTubeCaptionSender(bus, URL, transport=httpx.MockTransport(handle))
    await bus.publish(_line("one"))
    await bus.publish(_line("two"))
    with caplog.at_level("DEBUG"):
        task = asyncio.create_task(sender.run())
        async with asyncio.timeout(2):
            while sender.sent < 1:
                await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert str(seen[0].url) == URL + "&seq=1"
    assert seen[0].content == b"2026-10-25T17:00:00.250\none\n"
    assert "abc-123" not in caplog.text
    assert logging.getLogger("httpx").filters == []


async def test_stop_waits_for_the_sender_to_close() -> None:
    bus = EventBus()
    transport = httpx.MockTransport(lambda _request: httpx.Response(200))
    sender = YouTubeCaptionSender(bus, URL, transport=transport)
    slice_ = YouTubeCaptionsSlice(sender)
    await slice_.start()
    task = slice_._task
    await asyncio.sleep(0)
    await slice_.stop()
    assert task is not None and task.done()


async def test_full_queue_drops_the_oldest_line() -> None:
    bus = EventBus()
    fake = FakeYouTube()
    sender = YouTubeCaptionSender(bus, URL, post=fake.post, queue_size=2)
    for text in ("one", "two", "three"):
        await bus.publish(_line(text))
    await _run_until(fake, sender, 2)

    assert [c["body"].split("\n")[1] for c in fake.calls] == ["two", "three"]


def test_status_route_reports_disabled_without_a_url() -> None:
    app = create_app(settings=Settings(_env_file=None, youtube_caption_url=""))
    with TestClient(app) as client:
        assert client.get("/youtube-captions").json() == {"enabled": False}


def test_status_route_reports_counts_with_a_url() -> None:
    app = create_app(settings=Settings(_env_file=None, youtube_caption_url=URL))
    with TestClient(app) as client:
        body = client.get("/youtube-captions").json()
    assert body == {"enabled": True, "sent": 0, "failed": 0, "last_status": None}
