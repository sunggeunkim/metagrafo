from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from core.event_bus import EventBus
from core.settings import Settings
from features.operator_control.events import (
    CaptionsStateEvent,
    ModeChangedEvent,
    TranslateMode,
    languages_for,
)

_CONTROL_PAGE = Path(__file__).with_name("control.html")


class OperatorState:
    def __init__(self, mode: TranslateMode) -> None:
        self.captions_active = False
        self.mode = mode

    def snapshot(self) -> dict[str, object]:
        source, target = languages_for(self.mode)
        return {
            "captions_active": self.captions_active,
            "mode": self.mode.value,
            "source_language": source,
            "target_language": target,
        }


class CaptionsBody(BaseModel):
    is_active: bool


class ModeBody(BaseModel):
    mode: TranslateMode = Field(description="ko_to_en or en_to_en")


def register(app: FastAPI, bus: EventBus, settings: Settings) -> OperatorState:
    try:
        initial_mode = TranslateMode(settings.translate_mode)
    except ValueError:
        initial_mode = TranslateMode.KO_TO_EN
    state = OperatorState(initial_mode)
    app.state.operator = state

    @app.get("/control")
    async def control_page() -> object:
        from fastapi.responses import HTMLResponse

        return HTMLResponse(_CONTROL_PAGE.read_text(encoding="utf-8"))

    @app.get("/mode")
    async def get_mode() -> dict[str, object]:
        return state.snapshot()

    @app.put("/mode")
    async def put_mode(body: ModeBody) -> dict[str, object]:
        state.mode = body.mode
        source, target = languages_for(state.mode)
        await bus.publish(
            ModeChangedEvent(
                mode=state.mode.value,
                source_language=source,
                target_language=target,
            )
        )
        return state.snapshot()

    @app.put("/captions")
    async def put_captions(body: CaptionsBody) -> dict[str, object]:
        state.captions_active = body.is_active
        await bus.publish(CaptionsStateEvent(is_active=body.is_active))
        return state.snapshot()

    return state
