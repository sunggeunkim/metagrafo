from __future__ import annotations

import asyncio
import logging
from collections import deque

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from core.event_bus import EventBus
from features.operator_control.events import (
    CaptionsStateEvent,
    ModeChangedEvent,
    OneSentenceEvent,
    OverlayStyleEvent,
)
from features.translate_speech.events import SubtitleEvent

logger = logging.getLogger(__name__)


class SubtitleHub:
    def __init__(
        self,
        bus: EventBus,
        *,
        min_on_screen_s: float = 2.5,
        chars_per_s: float = 9.0,
        max_behind_s: float = 5.0,
        one_sentence: bool = True,
    ) -> None:
        self._clients: set[WebSocket] = set()
        self._min_on_screen_s = min_on_screen_s
        self._chars_per_s = chars_per_s
        self._max_behind_s = max_behind_s
        # Paced lines waiting for the one on screen to be read, oldest first.
        self._waiting: deque[dict[str, object]] = deque()
        self._shown_until = 0.0
        self._pacer: asyncio.Task[None] | None = None
        # Sent with the style, so an overlay learns it on connect and on every change.
        self._one_sentence = one_sentence
        # Captions always start on; see Settings.
        self._captions_active = True
        self._overlay_style = {
            "type": "overlay_style",
            "position": "top_left",
            "font_size_vw": 2.5,
            "inset_vertical_pct": 1.0,
            "inset_horizontal_pct": 1.0,
            "box_width_pct": 100.0,
            "box_height_pct": 30.0,
            "line_count": 3,
        }
        bus.subscribe(SubtitleEvent, self._on_subtitle)
        bus.subscribe(CaptionsStateEvent, self._on_captions)
        bus.subscribe(ModeChangedEvent, self._on_mode)
        bus.subscribe(OverlayStyleEvent, self._on_overlay_style)
        bus.subscribe(OneSentenceEvent, self._on_one_sentence)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        await websocket.send_json(self._style_message())
        # Lines stay up until replaced, so an overlay that reconnects after captions
        # stopped must be told, or it keeps showing its old lines.
        await websocket.send_json({"type": "captions_state", "is_active": self._captions_active})

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    async def _on_subtitle(self, event: SubtitleEvent) -> None:
        payload: dict[str, object] = {
            "type": "subtitle",
            "text": event.text,
            "chunk_id": event.chunk_id,
            "source_language": event.source_language,
            "target_language": event.target_language,
            "mode": event.mode,
            "ts": event.created_at,
            "duration_s": event.duration_s,
            "provisional": event.provisional,
        }
        if not event.paced:
            await self._broadcast(payload)
            return
        now = asyncio.get_running_loop().time()
        if not self._waiting and now >= self._shown_until:
            await self._show(payload)
            return
        self._waiting.append(payload)
        self._catch_up(now)
        if self._pacer is None:
            self._pacer = asyncio.create_task(self._pace(), name="subtitle-pacer")

    def _catch_up(self, now: float) -> None:
        """Drop the oldest waiting line while the newest would show more than max_behind_s late.

        Dropping, not merging: a merged line needs as long to read as its parts, so only
        dropping catches up. The newest line is never dropped, so it waits at most the rest
        of the line on screen.
        """
        behind = self._shown_until - now + sum(
            self._time_on_screen(waiting) for waiting in list(self._waiting)[:-1]
        )
        while behind > self._max_behind_s and len(self._waiting) > 1:
            behind -= self._time_on_screen(self._waiting.popleft())

    def _time_on_screen(self, payload: dict[str, object]) -> float:
        return max(self._min_on_screen_s, len(str(payload["text"])) / self._chars_per_s)

    async def _show(self, payload: dict[str, object]) -> None:
        on_screen = self._time_on_screen(payload)
        # Set before the send, so a line arriving mid-send waits its turn.
        self._shown_until = asyncio.get_running_loop().time() + on_screen
        # The overlay has no hide timer; the time is sent for anything that wants it.
        await self._broadcast({**payload, "min_on_screen_s": on_screen})

    async def _pace(self) -> None:
        loop = asyncio.get_running_loop()
        while self._waiting:
            await asyncio.sleep(max(0.0, self._shown_until - loop.time()))
            await self._show(self._waiting.popleft())
        self._pacer = None

    def _stop_pacing(self) -> None:
        """Captions off: the waiting lines go, and so does the pacer's wait, so none is sent later."""
        self._waiting.clear()
        # The overlay is cleared, so the next line need not wait for the old one to be read.
        self._shown_until = 0.0
        if self._pacer is not None:
            self._pacer.cancel()
            self._pacer = None

    async def _on_captions(self, event: CaptionsStateEvent) -> None:
        self._captions_active = event.is_active
        if not event.is_active:
            self._stop_pacing()
        await self._broadcast({"type": "captions_state", "is_active": event.is_active})

    async def _on_overlay_style(self, event: OverlayStyleEvent) -> None:
        self._overlay_style = {
            "type": "overlay_style",
            "position": event.position,
            "font_size_vw": event.font_size_vw,
            "inset_vertical_pct": event.inset_vertical_pct,
            "inset_horizontal_pct": event.inset_horizontal_pct,
            "box_width_pct": event.box_width_pct,
            "box_height_pct": event.box_height_pct,
            "line_count": event.line_count,
        }
        await self._broadcast(self._style_message())

    async def _on_one_sentence(self, event: OneSentenceEvent) -> None:
        self._one_sentence = event.one_sentence
        await self._broadcast(self._style_message())

    def _style_message(self) -> dict[str, object]:
        return {**self._overlay_style, "one_sentence": self._one_sentence}

    async def _on_mode(self, event: ModeChangedEvent) -> None:
        await self._broadcast(
            {
                "type": "mode",
                "mode": event.mode,
                "source_language": event.source_language,
                "target_language": event.target_language,
            }
        )

    async def _broadcast(self, payload: dict[str, object]) -> None:
        dead: list[WebSocket] = []
        for client in list(self._clients):
            if client.client_state != WebSocketState.CONNECTED:
                dead.append(client)
                continue
            try:
                await client.send_json(payload)
            except (WebSocketDisconnect, RuntimeError) as exc:
                logger.info("dropping websocket client: %s", exc)
                dead.append(client)
        for client in dead:
            self._clients.discard(client)
