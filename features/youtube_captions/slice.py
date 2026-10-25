from __future__ import annotations

import asyncio

from fastapi import FastAPI

from core.event_bus import EventBus
from core.settings import Settings
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
            self._task = None


def register(app: FastAPI, bus: EventBus, settings: Settings, *, post=None) -> YouTubeCaptionsSlice:
    url = settings.youtube_caption_url.strip()
    sender = YouTubeCaptionSender(bus, url, post=post) if url else None

    @app.get("/youtube-captions")
    async def youtube_captions() -> dict[str, object]:
        if sender is None:
            return {"enabled": False}
        return sender.snapshot()

    return YouTubeCaptionsSlice(sender)
