from __future__ import annotations

import logging

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from core.event_bus import EventBus
from features.operator_control.events import (
    CaptionsStateEvent,
    ModeChangedEvent,
    OverlayStyleEvent,
)
from features.translate_speech.events import SubtitleEvent

logger = logging.getLogger(__name__)


class SubtitleHub:
    def __init__(self, bus: EventBus) -> None:
        self._clients: set[WebSocket] = set()
        self._overlay_style = {
            "type": "overlay_style",
            "position": "top_left",
            "font_size_vw": 2.5,
            "inset_vertical_pct": 1.0,
            "inset_horizontal_pct": 1.0,
            "box_width_pct": 100.0,
            "box_height_pct": 10.0,
        }
        bus.subscribe(SubtitleEvent, self._on_subtitle)
        bus.subscribe(CaptionsStateEvent, self._on_captions)
        bus.subscribe(ModeChangedEvent, self._on_mode)
        bus.subscribe(OverlayStyleEvent, self._on_overlay_style)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        await websocket.send_json(self._overlay_style)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    async def _on_subtitle(self, event: SubtitleEvent) -> None:
        await self._broadcast(
            {
                "type": "subtitle",
                "text": event.text,
                "chunk_id": event.chunk_id,
                "source_language": event.source_language,
                "target_language": event.target_language,
                "mode": event.mode,
                "ts": event.created_at,
                "duration_s": event.duration_s,
            }
        )

    async def _on_captions(self, event: CaptionsStateEvent) -> None:
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
        }
        await self._broadcast(self._overlay_style)

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
