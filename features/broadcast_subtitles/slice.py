from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from core.event_bus import EventBus
from core.settings import Settings
from features.broadcast_subtitles.hub import SubtitleHub
from features.operator_control.events import languages_for
from features.translate_speech.events import SubtitleEvent

_OVERLAY_PAGE = Path(__file__).with_name("overlay.html")


class InjectBody(BaseModel):
    text: str = Field(min_length=1)


def register(app: FastAPI, bus: EventBus, settings: Settings) -> SubtitleHub:
    hub = SubtitleHub(bus)
    app.state.hub = hub

    @app.get("/overlay")
    async def overlay() -> HTMLResponse:
        return HTMLResponse(_OVERLAY_PAGE.read_text(encoding="utf-8"))

    @app.get("/health")
    async def health() -> dict[str, object]:
        operator = app.state.operator
        profile = getattr(app.state, "whisper_profile", None)
        body: dict[str, object] = {
            "ok": True,
            "clients": hub.client_count,
            **operator.snapshot(),
        }
        if profile is not None:
            body.update(
                {
                    "model": profile.model,
                    "device": profile.device,
                    "compute_type": profile.compute_type,
                    "vram_gb": profile.vram_gb,
                }
            )
        return body

    @app.post("/inject")
    async def inject(body: InjectBody) -> dict[str, str]:
        operator = app.state.operator
        source, target = languages_for(operator.mode)
        chunk_id = str(uuid.uuid4())
        await bus.publish(
            SubtitleEvent(
                chunk_id=chunk_id,
                text=body.text.strip(),
                source_language=source,
                target_language=target,
                mode=operator.mode.value,
                created_at=time.time(),
                duration_s=0.0,
            )
        )
        return {"chunk_id": chunk_id}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await hub.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            hub.disconnect(websocket)

    return hub
