from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from core.event_bus import EventBus
from core.settings import Settings
from features.operator_control.events import (
    CaptionsStateEvent,
    ModeChangedEvent,
    OverlayPosition,
    OverlayStyleEvent,
    TranslateMode,
    VadSilenceMsEvent,
    languages_for,
)

_CONTROL_PAGE = Path(__file__).with_name("control.html")


class OperatorState:
    def __init__(
        self,
        mode: TranslateMode,
        *,
        vad_min_silence_ms: int = 500,
        overlay_position: OverlayPosition = OverlayPosition.TOP_LEFT,
        font_size_vw: float = 2.5,
        inset_vertical_pct: float = 1.0,
        inset_horizontal_pct: float = 1.0,
        box_width_pct: float = 100.0,
        box_height_pct: float = 30.0,
    ) -> None:
        self.captions_active = True
        self.mode = mode
        self.vad_min_silence_ms = vad_min_silence_ms
        self.overlay_position = overlay_position
        self.font_size_vw = font_size_vw
        self.inset_vertical_pct = inset_vertical_pct
        self.inset_horizontal_pct = inset_horizontal_pct
        self.box_width_pct = box_width_pct
        self.box_height_pct = box_height_pct

    def snapshot(self) -> dict[str, object]:
        source, target = languages_for(self.mode)
        return {
            "captions_active": self.captions_active,
            "mode": self.mode.value,
            "source_language": source,
            "target_language": target,
            "vad_min_silence_ms": self.vad_min_silence_ms,
            "position": self.overlay_position.value,
            "font_size_vw": self.font_size_vw,
            "inset_vertical_pct": self.inset_vertical_pct,
            "inset_horizontal_pct": self.inset_horizontal_pct,
            "box_width_pct": self.box_width_pct,
            "box_height_pct": self.box_height_pct,
        }


class CaptionsBody(BaseModel):
    is_active: bool


class ModeBody(BaseModel):
    mode: TranslateMode = Field(description="ko_to_en or en_to_en")


class VadSilenceBody(BaseModel):
    vad_min_silence_ms: int = Field(ge=100, le=3000)


class OverlayStyleBody(BaseModel):
    position: OverlayPosition
    font_size_vw: float = Field(ge=1.0, le=8.0)
    inset_vertical_pct: float = Field(ge=0.0, le=20.0)
    inset_horizontal_pct: float = Field(ge=0.0, le=20.0)
    box_width_pct: float = Field(ge=10.0, le=100.0)
    box_height_pct: float = Field(ge=5.0, le=100.0)


def register(app: FastAPI, bus: EventBus, settings: Settings) -> OperatorState:
    try:
        initial_mode = TranslateMode(settings.translate_mode)
    except ValueError:
        initial_mode = TranslateMode.KO_TO_EN
    state = OperatorState(
        initial_mode,
        vad_min_silence_ms=settings.vad_min_silence_ms,
    )
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

    @app.put("/vad-silence")
    async def put_vad_silence(body: VadSilenceBody) -> dict[str, object]:
        state.vad_min_silence_ms = body.vad_min_silence_ms
        await bus.publish(VadSilenceMsEvent(vad_min_silence_ms=body.vad_min_silence_ms))
        return state.snapshot()

    @app.put("/overlay-style")
    async def put_overlay_style(body: OverlayStyleBody) -> dict[str, object]:
        state.overlay_position = body.position
        state.font_size_vw = body.font_size_vw
        state.inset_vertical_pct = body.inset_vertical_pct
        state.inset_horizontal_pct = body.inset_horizontal_pct
        state.box_width_pct = body.box_width_pct
        state.box_height_pct = body.box_height_pct
        await bus.publish(
            OverlayStyleEvent(
                position=body.position.value,
                font_size_vw=body.font_size_vw,
                inset_vertical_pct=body.inset_vertical_pct,
                inset_horizontal_pct=body.inset_horizontal_pct,
                box_width_pct=body.box_width_pct,
                box_height_pct=body.box_height_pct,
            )
        )
        return state.snapshot()

    return state
