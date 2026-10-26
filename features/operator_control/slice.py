from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.event_bus import EventBus
from core.settings import Settings
from features.operator_control.events import (
    AudioDeviceChangedEvent,
    CaptionsStateEvent,
    GeminiKeyChangedEvent,
    ModeChangedEvent,
    OverlayPosition,
    OverlayStyleEvent,
    TranslateMode,
    ConditionOnPreviousTextEvent,
    VadPausesEvent,
    VadSilenceMsEvent,
    YouTubeCaptionUrlChangedEvent,
    languages_for,
    needs_gemini,
)

_CONTROL_PAGE = Path(__file__).with_name("control.html")


class OperatorState:
    def __init__(
        self,
        mode: TranslateMode,
        *,
        vad_min_silence_ms: int = 500,
        vad_pauses: int = 1,
        condition_on_previous_text: bool = True,
        overlay_position: OverlayPosition = OverlayPosition.TOP_LEFT,
        font_size_vw: float = 2.5,
        inset_vertical_pct: float = 1.0,
        inset_horizontal_pct: float = 1.0,
        box_width_pct: float = 100.0,
        box_height_pct: float = 30.0,
        line_count: int = 2,
    ) -> None:
        self.captions_active = True
        self.mode = mode
        self.vad_min_silence_ms = vad_min_silence_ms
        self.vad_pauses = vad_pauses
        self.condition_on_previous_text = condition_on_previous_text
        self.overlay_position = overlay_position
        self.font_size_vw = font_size_vw
        self.inset_vertical_pct = inset_vertical_pct
        self.inset_horizontal_pct = inset_horizontal_pct
        self.box_width_pct = box_width_pct
        self.box_height_pct = box_height_pct
        self.line_count = line_count

    def snapshot(self) -> dict[str, object]:
        source, target = languages_for(self.mode)
        return {
            "captions_active": self.captions_active,
            "mode": self.mode.value,
            "source_language": source,
            "target_language": target,
            "vad_min_silence_ms": self.vad_min_silence_ms,
            "vad_pauses": self.vad_pauses,
            "condition_on_previous_text": self.condition_on_previous_text,
            "position": self.overlay_position.value,
            "font_size_vw": self.font_size_vw,
            "inset_vertical_pct": self.inset_vertical_pct,
            "inset_horizontal_pct": self.inset_horizontal_pct,
            "box_width_pct": self.box_width_pct,
            "box_height_pct": self.box_height_pct,
            "line_count": self.line_count,
        }


class GeminiKeyBody(BaseModel):
    api_key: str


class CaptionsBody(BaseModel):
    is_active: bool


class ModeBody(BaseModel):
    mode: TranslateMode = Field(description="ko_to_en, en_to_en, or en_to_ko")


class VadSilenceBody(BaseModel):
    vad_min_silence_ms: int = Field(ge=100, le=3000)


class VadPausesBody(BaseModel):
    vad_pauses: int = Field(ge=1, le=2)


class ConditionOnPreviousTextBody(BaseModel):
    condition_on_previous_text: bool


class OverlayStyleBody(BaseModel):
    position: OverlayPosition
    font_size_vw: float = Field(ge=1.0, le=8.0)
    inset_vertical_pct: float = Field(ge=0.0, le=20.0)
    inset_horizontal_pct: float = Field(ge=0.0, le=20.0)
    box_width_pct: float = Field(ge=10.0, le=100.0)
    box_height_pct: float = Field(ge=5.0, le=100.0)
    line_count: int = Field(default=2, ge=1, le=6)


class AudioDeviceBody(BaseModel):
    name: str = Field(description="One input name from GET /audio-devices")


class YouTubeCaptionUrlBody(BaseModel):
    url: str = Field(description="Captions Ingestion URL from YouTube Studio; empty stops sending")


def write_env_values(env_file: Path, values: dict[str, str]) -> None:
    """Set each NAME=value in .env, replacing the old line and keeping every other line."""
    existing = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
    written: set[str] = set()
    lines: list[str] = []
    for line in existing:
        name = line.strip().split("=", 1)[0]
        if "=" in line and name in values:
            lines.append(f"{name}={values[name]}")
            written.add(name)
        else:
            lines.append(line)
    lines.extend(f"{name}={value}" for name, value in values.items() if name not in written)
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gemini_key(env_file: Path, api_key: str) -> None:
    write_env_values(env_file, {"CAPTION_ENGINE": "gemini_live", "GEMINI_API_KEY": api_key})


def _has_line_break(text: str) -> bool:
    return "\n" in text or "\r" in text


def is_caption_url(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme in ("http", "https") and bool(parts.netloc) and not any(
        ch.isspace() for ch in url
    )


def register(
    app: FastAPI,
    bus: EventBus,
    settings: Settings,
    *,
    env_file: Path | None = None,
    list_input_devices: Callable[[], list[str]] | None = None,
    capture_status: Callable[[], dict[str, str | None]] | None = None,
) -> OperatorState:
    try:
        initial_mode = TranslateMode(settings.translate_mode)
    except ValueError:
        initial_mode = TranslateMode.KO_TO_EN
    if needs_gemini(initial_mode) and not settings.gemini_live():
        initial_mode = TranslateMode.KO_TO_EN
    # A key pasted later starts Gemini from this setting, so it must match the booth.
    settings.translate_mode = initial_mode.value
    pauses = settings.vad_pauses if settings.vad_pauses in (1, 2) else 1
    key_file = env_file or Path.cwd() / ".env"
    state = OperatorState(
        initial_mode,
        vad_min_silence_ms=settings.vad_min_silence_ms,
        vad_pauses=pauses,
        condition_on_previous_text=settings.condition_on_previous_text,
    )
    app.state.operator = state

    def public_state() -> dict[str, object]:
        body = state.snapshot()
        body["gemini_key_set"] = bool(settings.gemini_api_key.strip())
        body["gemini_live"] = settings.gemini_live()
        body["youtube_url_set"] = bool(settings.youtube_caption_url.strip())
        body["audio_device"] = settings.audio_device_name
        status = capture_status() if capture_status is not None else {}
        body["audio_capturing"] = status.get("capturing")
        body["audio_error"] = status.get("error")
        return body

    async def input_names() -> list[str]:
        if list_input_devices is None:
            return []
        # Opening PortAudio to enumerate takes a moment; keep it off the event loop.
        names = await asyncio.to_thread(list_input_devices)
        # Some Windows Bluetooth names carry a line break; one line of .env cannot hold them.
        return [n for n in names if not _has_line_break(n)]

    def audio_devices(names: list[str]) -> dict[str, object]:
        status = capture_status() if capture_status is not None else {}
        return {
            "devices": names,
            "selected": settings.audio_device_name,
            "capturing": status.get("capturing"),
            "error": status.get("error"),
        }


    @app.get("/control")
    async def control_page() -> object:
        from fastapi.responses import HTMLResponse

        return HTMLResponse(_CONTROL_PAGE.read_text(encoding="utf-8"))

    @app.get("/mode")
    async def get_mode() -> dict[str, object]:
        return public_state()

    @app.put("/gemini-key")
    async def put_gemini_key(body: GeminiKeyBody) -> dict[str, object]:
        key = body.api_key.strip()
        if not key or "\n" in body.api_key or "\r" in body.api_key:
            raise HTTPException(status_code=422, detail="Gemini key is empty")
        write_gemini_key(key_file, key)
        settings.gemini_api_key = key
        settings.caption_engine = "gemini_live"
        await bus.publish(GeminiKeyChangedEvent())
        return public_state()

    @app.put("/youtube-caption-url")
    async def put_youtube_caption_url(body: YouTubeCaptionUrlBody) -> dict[str, object]:
        # The URL carries the stream's caption key, so it is never echoed back.
        url = body.url.strip()
        if url and not is_caption_url(url):
            raise HTTPException(
                status_code=422,
                detail="Paste the Captions Ingestion URL from YouTube Studio (http://upload.youtube.com/closedcaption?cid=…).",
            )
        write_env_values(key_file, {"YOUTUBE_CAPTION_URL": url})
        settings.youtube_caption_url = url
        await bus.publish(YouTubeCaptionUrlChangedEvent())
        return public_state()

    @app.get("/audio-devices")
    async def get_audio_devices() -> dict[str, object]:
        return audio_devices(await input_names())

    @app.put("/audio-device")
    async def put_audio_device(body: AudioDeviceBody) -> dict[str, object]:
        name = body.name.strip()
        if settings.audio_device_index is not None:
            raise HTTPException(
                status_code=409,
                detail="AUDIO_DEVICE_INDEX is set in .env; remove it to choose by name.",
            )
        # The name goes into .env, so only a name this PC lists is accepted.
        names = await input_names()
        if not name or _has_line_break(name) or name not in names:
            raise HTTPException(status_code=422, detail=f"No input named {name!r} on this PC.")
        write_env_values(key_file, {"AUDIO_DEVICE_NAME": name})
        settings.audio_device_name = name
        await bus.publish(AudioDeviceChangedEvent())
        # Reuse the list: enumerating again would race capture reopening PortAudio.
        return audio_devices(names)

    @app.put("/mode")
    async def put_mode(body: ModeBody) -> dict[str, object]:
        if needs_gemini(body.mode) and not settings.gemini_live():
            raise HTTPException(
                status_code=409,
                detail="Korean captions need Gemini. Save a Gemini key first.",
            )
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

    @app.put("/vad-pauses")
    async def put_vad_pauses(body: VadPausesBody) -> dict[str, object]:
        state.vad_pauses = body.vad_pauses
        await bus.publish(VadPausesEvent(vad_pauses=body.vad_pauses))
        return state.snapshot()

    @app.put("/condition-on-previous-text")
    async def put_condition_on_previous_text(
        body: ConditionOnPreviousTextBody,
    ) -> dict[str, object]:
        state.condition_on_previous_text = body.condition_on_previous_text
        await bus.publish(
            ConditionOnPreviousTextEvent(
                condition_on_previous_text=body.condition_on_previous_text
            )
        )
        return state.snapshot()

    @app.put("/overlay-style")
    async def put_overlay_style(body: OverlayStyleBody) -> dict[str, object]:
        state.overlay_position = body.position
        state.font_size_vw = body.font_size_vw
        state.inset_vertical_pct = body.inset_vertical_pct
        state.inset_horizontal_pct = body.inset_horizontal_pct
        state.box_width_pct = body.box_width_pct
        state.box_height_pct = body.box_height_pct
        state.line_count = body.line_count
        await bus.publish(
            OverlayStyleEvent(
                position=body.position.value,
                font_size_vw=body.font_size_vw,
                inset_vertical_pct=body.inset_vertical_pct,
                inset_horizontal_pct=body.inset_horizontal_pct,
                box_width_pct=body.box_width_pct,
                box_height_pct=body.box_height_pct,
                line_count=body.line_count,
            )
        )
        return state.snapshot()

    return state
