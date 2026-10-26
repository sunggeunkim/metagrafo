from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx

from core.event_bus import EventBus
from features.translate_speech.events import SubtitleEvent

logger = logging.getLogger(__name__)


def format_timestamp(epoch_s: float) -> str:
    """YouTube wants UTC with milliseconds and no zone suffix: 2026-10-25T17:00:00.000."""
    dt = datetime.fromtimestamp(epoch_s, tz=UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def caption_body(epoch_s: float, text: str) -> str:
    return f"{format_timestamp(epoch_s)}\n{' '.join(text.split())}\n"


def sequenced_url(url: str, seq: int) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}seq={seq}"


class _HideCaptionKey(logging.Filter):
    """httpx logs every request URL at INFO, and this URL carries the stream's caption key."""

    def __init__(self, current_url: Callable[[], str]) -> None:
        super().__init__()
        self._current_url = current_url

    def filter(self, record: logging.LogRecord) -> bool:
        url = self._current_url()
        secret = urlsplit(url).query or url
        return not secret or secret not in record.getMessage()


class YouTubeCaptionSender:
    """Posts each finished caption line to a YouTube live caption ingestion URL.

    Lines are sent one at a time, in order, so `seq` always rises. The bottom line
    that Gemini live is still growing (`provisional`) is not sent. With no URL,
    lines are dropped until `use_url` gives one.
    """

    def __init__(
        self,
        bus: EventBus,
        url: str,
        *,
        queue_size: int = 16,
        timeout_s: float = 10.0,
        retry_delays_s: tuple[float, ...] = (0.5, 1.0),
        transport: httpx.AsyncBaseTransport | None = None,
        post=None,
    ) -> None:
        self._url = url.strip()
        self._queue: asyncio.Queue[tuple[float, str]] = asyncio.Queue(maxsize=queue_size)
        self._timeout_s = timeout_s
        self._retry_delays_s = retry_delays_s
        self._transport = transport
        self._post = post
        self._seq = 1
        self.sent = 0
        self.failed = 0
        self.last_status: int | None = None
        self._failing = False
        bus.subscribe(SubtitleEvent, self._on_subtitle)

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    def use_url(self, url: str) -> None:
        """Send to a new URL (a new stream), or stop sending when it is empty."""
        url = url.strip()
        if url == self._url:
            return
        self._url = url
        # Each ingestion URL counts its own seq from 1.
        self._seq = 1
        self.sent = 0
        self.failed = 0
        self.last_status = None
        self._failing = False
        while not self._queue.empty():
            self._queue.get_nowait()

    def snapshot(self) -> dict[str, object]:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "sent": self.sent,
            "failed": self.failed,
            "last_status": self.last_status,
        }

    async def _on_subtitle(self, event: SubtitleEvent) -> None:
        if not self._url or event.provisional or not event.text.strip():
            return
        # Stamp the line when the speech began, so YouTube shows it with the words.
        spoken_at = event.spoken_at if event.spoken_at is not None else event.created_at
        if self._queue.full():
            self._queue.get_nowait()
            logger.warning("youtube caption queue full; dropped the oldest line")
        self._queue.put_nowait((spoken_at, event.text))

    async def run(self) -> None:
        if self._post is not None:
            await self._drain(self._post)
            return
        hide = _HideCaptionKey(lambda: self._url)
        httpx_logger = logging.getLogger("httpx")
        httpx_logger.addFilter(hide)
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_s, transport=self._transport
            ) as client:
                await self._drain(client.post)
        finally:
            httpx_logger.removeFilter(hide)

    async def _drain(self, post) -> None:
        while True:
            spoken_at, text = await self._queue.get()
            await self._send(post, caption_body(spoken_at, text))

    async def _send(self, post, body: str) -> None:
        # The URL carries the stream's caption key, so it is never logged.
        url = self._url
        for delay_s in (*self._retry_delays_s, None):
            try:
                response = await post(
                    sequenced_url(url, self._seq),
                    content=body.encode("utf-8"),
                    headers={"Content-Type": "text/plain"},
                )
            except httpx.TransportError as exc:
                # YouTube may have taken the line before the connection broke, so the
                # same line goes again with the same seq.
                if delay_s is not None:
                    await asyncio.sleep(delay_s)
                    if self._url != url:
                        return
                    continue
                if self._url != url:
                    return
                # Out of retries. A skipped seq is safe; a reused one can hide the next line.
                self._seq += 1
                self.last_status = None
                self._record_failure(f"request failed: {type(exc).__name__}")
                return
            except httpx.HTTPError as exc:
                self.last_status = None
                self._record_failure(f"request failed: {type(exc).__name__}")
                return
            break
        if self._url != url:
            # The operator switched streams mid-line; the new stream's counts start clean.
            return
        self.last_status = response.status_code
        if 200 <= response.status_code < 300:
            self._seq += 1
            self.sent += 1
            if self._failing:
                logger.info("youtube captions are being accepted again")
                self._failing = False
            return
        # A refusal (stream not live yet) keeps seq, so the first accepted line is seq=1.
        self._record_failure(f"returned {response.status_code}")

    def _record_failure(self, reason: str) -> None:
        self.failed += 1
        # YouTube refuses lines until the stream is live; warn once, not every line.
        if not self._failing:
            logger.warning("youtube captions %s; is the stream live?", reason)
            self._failing = True
        else:
            logger.debug("youtube captions %s", reason)
