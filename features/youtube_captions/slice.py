from __future__ import annotations

import asyncio
import contextlib

from fastapi import FastAPI

from core.event_bus import EventBus
from core.settings import Settings
from features.operator_control.events import YouTubeCaptionUrlChangedEvent
from features.youtube_captions.sender import YouTubeCaptionSender


class YouTubeCaptionsSlice:
    def __init__(self, sender: YouTubeCaptionSender | None = None) -> None:
        self.sender = sender
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self.sender is not None:
            self._task = asyncio.create_task(self.sender.run(), name="youtube-captions")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            # Wait so the HTTP client closes before the loop shuts down.
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None


def register(app: FastAPI, bus: EventBus, settings: Settings, *, post=None) -> YouTubeCaptionsSlice:
    # The sender runs even without a URL, so one saved on /control starts sending at once.
    sender = YouTubeCaptionSender(bus, settings.youtube_caption_url, post=post)

    async def on_url_changed(_event: YouTubeCaptionUrlChangedEvent) -> None:
        sender.use_url(settings.youtube_caption_url)

    bus.subscribe(YouTubeCaptionUrlChangedEvent, on_url_changed)

    @app.get("/youtube-captions")
    async def youtube_captions() -> dict[str, object]:
        return sender.snapshot()

    return YouTubeCaptionsSlice(sender)
