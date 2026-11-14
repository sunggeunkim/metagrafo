from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from core.event_bus import EventBus
from core.settings import ENGINE_NAMES, GLOSSARY_DEFAULT, HERMENEIA_MODELS, KOREAN_STYLE_DEFAULT, Settings
from features.operator_control.events import (
    AudioDeviceChangedEvent,
    CaptionEngineChangedEvent,
    CaptionsStateEvent,
    CleanUpWordingEvent,
    CleanupContextEvent,
    GeminiKeyChangedEvent,
    GlossaryEvent,
    KoreanStyleEvent,
    ModeChangedEvent,
    OneSentenceEvent,
    SaveTranscriptEvent,
    OverlayPosition,
    OverlayStyleEvent,
    TranslateMode,
    ConditionOnPreviousTextEvent,
    VadPausesEvent,
    VadSilenceMsEvent,
    WholeSentencesEvent,
    YouTubeCaptionUrlChangedEvent,
    languages_for,
    needs_gemini,
)

_CONTROL_PAGE = Path(__file__).with_name("control.html")
# The additional context goes with every cleaned line, so it stays short enough not to slow cleanup.
MAX_CLEANUP_CONTEXT_CHARS = 2000
# The Korean style also goes with every cleaned line; a style needs far less than the context.
MAX_KOREAN_STYLE_CHARS = 1000
# The glossary goes with every cleaned line too; about 3000 characters is under 2000 tokens.
MAX_GLOSSARY_CHARS = 3000

logger = logging.getLogger(__name__)


class OperatorState:
    def __init__(
        self,
        mode: TranslateMode,
        *,
        vad_min_silence_ms: int = 500,
        vad_pauses: int = 1,
        condition_on_previous_text: bool = True,
        whole_sentences: bool = True,
        clean_up_wording: bool = True,
        save_transcript: bool = False,
        one_sentence: bool = True,
        overlay_position: OverlayPosition = OverlayPosition.TOP_LEFT,
        font_size_vw: float = 2.5,
        inset_vertical_pct: float = 1.0,
        inset_horizontal_pct: float = 1.0,
        box_width_pct: float = 100.0,
        box_height_pct: float = 30.0,
        line_count: int = 3,
    ) -> None:
        self.captions_active = True
        self.mode = mode
        self.vad_min_silence_ms = vad_min_silence_ms
        self.vad_pauses = vad_pauses
        self.condition_on_previous_text = condition_on_previous_text
        self.whole_sentences = whole_sentences
        self.clean_up_wording = clean_up_wording
        self.save_transcript = save_transcript
        self.one_sentence = one_sentence
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
            "whole_sentences": self.whole_sentences,
            "clean_up_wording": self.clean_up_wording,
            "save_transcript": self.save_transcript,
            "one_sentence": self.one_sentence,
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


class HermeneiaTokenBody(BaseModel):
    token: str


class CaptionEngineBody(BaseModel):
    engine: Literal["gemini_live", "whisper_local", "whisper_hermeneia", "seamless_hermeneia"] = Field(
        description="gemini_live, Whisper on this PC, or Whisper or Seamless on the Hermeneia server"
    )


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


class WholeSentencesBody(BaseModel):
    whole_sentences: bool


class OneSentenceBody(BaseModel):
    one_sentence: bool


class CleanUpWordingBody(BaseModel):
    clean_up_wording: bool


class CleanupContextBody(BaseModel):
    text: str = Field(max_length=MAX_CLEANUP_CONTEXT_CHARS)


class KoreanStyleBody(BaseModel):
    text: str = Field(max_length=MAX_KOREAN_STYLE_CHARS)


class GlossaryBody(BaseModel):
    text: str = Field(max_length=MAX_GLOSSARY_CHARS)


class SaveTranscriptBody(BaseModel):
    save_transcript: bool


class OverlayStyleBody(BaseModel):
    position: OverlayPosition
    font_size_vw: float = Field(ge=1.0, le=8.0)
    inset_vertical_pct: float = Field(ge=0.0, le=20.0)
    inset_horizontal_pct: float = Field(ge=0.0, le=20.0)
    box_width_pct: float = Field(ge=10.0, le=100.0)
    box_height_pct: float = Field(ge=5.0, le=100.0)
    line_count: int = Field(default=3, ge=1, le=6)


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


def _read_booth_text(path: Path, what: str, limit: int) -> str | None:
    """Text booth control saved in `path`, or None when there is no file or it cannot be read.

    The file is the only source: booth control writes nothing else. A hand-edited file is
    held to the same limit as a save, and one that cannot be read never stops Metagrafo.
    """
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("%s not loaded: cannot read %s (%s)", what, path.name, exc)
        return None
    if len(text) > limit:
        logger.warning("%s in %s is %d characters; using the first %d", what, path.name, len(text), limit)
        text = text[:limit].strip()
    return text


def read_glossary(glossary_file: Path) -> str:
    """The saved glossary ("" turns it off), or the church default when none was saved."""
    text = _read_booth_text(glossary_file, "glossary", MAX_GLOSSARY_CHARS)
    return text if text is not None else GLOSSARY_DEFAULT


def read_korean_style(style_file: Path) -> str:
    """The saved Korean style ("" turns the rule off), or the default when none was saved."""
    text = _read_booth_text(style_file, "Korean style", MAX_KOREAN_STYLE_CHARS)
    return text if text is not None else KOREAN_STYLE_DEFAULT


def write_gemini_key(env_file: Path, api_key: str) -> None:
    write_env_values(env_file, {"CAPTION_ENGINE": "gemini_live", "GEMINI_API_KEY": api_key})


def read_log(log_file: Path, after: int | None = None, max_bytes: int = 256_000) -> tuple[str, int]:
    """Whole lines from byte `after` on, or the end of the log, and where the next read starts.

    Every start appends to the log and nothing trims it, so a first read takes only the end.
    """
    try:
        with log_file.open("rb") as log:
            size = log.seek(0, 2)
            # An offset past the end means the file was replaced; read it afresh.
            start = after if after is not None and after <= size else 0
            cut = size - start > max_bytes
            if cut:
                start = size - max_bytes
            log.seek(start)
            data = log.read(size - start)
    except FileNotFoundError:
        return "", 0
    if cut:
        # The cut lands mid-line; start at the next whole one.
        skip = data.find(b"\n") + 1
        start += skip
        data = data[skip:]
    # A line still being written waits for the next read.
    end = data.rfind(b"\n") + 1
    return data[:end].decode("utf-8", errors="replace"), start + end


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
    log_file: Path | None = None,
    request_quit: Callable[[], None] | None = None,
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
    # Notes about one service, kept only while Metagrafo runs: nobody remembers to clear them
    # for the next one. Replaces any CLEANUP_CONTEXT from the environment or .env.
    settings.cleanup_context = ""
    # No file means the default style; an empty file means the booth turned the rule off.
    style_file = key_file.with_name("korean-style.txt")
    settings.korean_style = read_korean_style(style_file)
    # Like the Korean style: no file means the default, an empty file means no glossary.
    glossary_file = key_file.with_name("glossary.txt")
    settings.glossary = read_glossary(glossary_file)
    state = OperatorState(
        initial_mode,
        vad_min_silence_ms=settings.vad_min_silence_ms,
        vad_pauses=pauses,
        condition_on_previous_text=settings.condition_on_previous_text,
        whole_sentences=settings.whole_sentences,
        clean_up_wording=settings.clean_up_wording,
        save_transcript=settings.save_transcript,
        one_sentence=settings.one_sentence,
    )
    app.state.operator = state
    # Counts engine picks (a Gemini key saved is one), so a pick that waited on Whisper loading
    # can tell a newer one landed meanwhile and leave it standing.
    engine_picks = 0

    def whisper_local_ready() -> bool:
        """Whisper is on this PC: loaded, or installed and loads when picked."""
        translator = getattr(app.state, "translator", None)
        return bool(getattr(translator, "whisper_local_ready", False))

    def public_state() -> dict[str, object]:
        body = state.snapshot()
        body["gemini_key_set"] = bool(settings.gemini_api_key.strip())
        body["gemini_live"] = settings.gemini_live()
        body["caption_engine"] = settings.engine()
        body["hermeneia_token_set"] = bool(settings.hermeneia_token.strip())
        body["whisper_local_ready"] = whisper_local_ready()
        body["youtube_url_set"] = bool(settings.youtube_caption_url.strip())
        body["cleanup_context"] = settings.cleanup_context
        body["korean_style"] = settings.korean_style
        body["korean_style_default"] = KOREAN_STYLE_DEFAULT
        body["glossary"] = settings.glossary
        body["glossary_default"] = GLOSSARY_DEFAULT
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

    @app.get("/logs")
    async def get_logs(after: int | None = Query(default=None, ge=0)) -> dict[str, object]:
        # A copy started in a console logs there; an old metagrafo.log would mislead.
        if log_file is None:
            return {"available": False, "text": "", "offset": 0}
        text, offset = await asyncio.to_thread(read_log, log_file, after)
        return {"available": True, "text": text, "offset": offset}

    @app.post("/quit")
    async def post_quit() -> dict[str, object]:
        # The server stops after this reply is sent, closing capture and Gemini first.
        if request_quit is None:
            raise HTTPException(
                status_code=409,
                detail="This Metagrafo was started from a console. Press Ctrl+C there to quit.",
            )
        request_quit()
        return {"quitting": True}

    @app.get("/mode")
    async def get_mode() -> dict[str, object]:
        return public_state()

    @app.put("/gemini-key")
    async def put_gemini_key(body: GeminiKeyBody) -> dict[str, object]:
        key = body.api_key.strip()
        if not key or "\n" in body.api_key or "\r" in body.api_key:
            raise HTTPException(status_code=422, detail="Gemini key is empty")
        nonlocal engine_picks
        engine_picks += 1
        write_gemini_key(key_file, key)
        settings.gemini_api_key = key
        settings.caption_engine = "gemini_live"
        await bus.publish(GeminiKeyChangedEvent())
        return public_state()

    @app.put("/hermeneia-token")
    async def put_hermeneia_token(body: HermeneiaTokenBody) -> dict[str, object]:
        # Like the Gemini key, the token is saved beside the program and never sent back.
        token = body.token.strip()
        if not token or _has_line_break(body.token):
            raise HTTPException(status_code=422, detail="Hermeneia token is empty")
        write_env_values(key_file, {"HERMENEIA_TOKEN": token})
        settings.hermeneia_token = token
        await bus.publish(CaptionEngineChangedEvent())
        return public_state()

    @app.put("/caption-engine")
    async def put_caption_engine(body: CaptionEngineBody) -> dict[str, object]:
        nonlocal engine_picks
        engine_picks += 1
        pick = engine_picks

        def refuse_in_korean() -> None:
            if needs_gemini(state.mode):
                raise HTTPException(
                    status_code=409,
                    detail="Only Gemini live writes Korean captions. Pick another mode first.",
                )

        if body.engine == "gemini_live":
            if not settings.gemini_api_key.strip():
                raise HTTPException(status_code=409, detail="Gemini needs a key. Save a Gemini key first.")
            values = {"CAPTION_ENGINE": "gemini_live"}
        else:
            refuse_in_korean()
            if body.engine == "whisper_local":
                # Loaded before the switch, so a GPU that cannot hold it leaves the engine as it was.
                translator = getattr(app.state, "translator", None)
                if translator is None:
                    raise HTTPException(status_code=409, detail="Whisper is not installed on this PC.")
                problem = await translator.load_local_whisper()
                if problem is not None:
                    raise HTTPException(status_code=409, detail=problem)
                # The load can take a minute, and the booth kept working meanwhile.
                if engine_picks != pick:
                    raise HTTPException(
                        status_code=409,
                        detail=f"{ENGINE_NAMES[settings.engine()]} was picked while Whisper loaded on this PC, "
                        "so it stays. Whisper on this PC is ready when you pick it again.",
                    )
                refuse_in_korean()
                values = {"CAPTION_ENGINE": "whisper", "USE_HERMENEIA": "false"}
            else:
                if not settings.hermeneia_token.strip():
                    raise HTTPException(
                        status_code=409,
                        detail="Hermeneia needs its token. Save the Hermeneia token first.",
                    )
                values = {
                    "CAPTION_ENGINE": "whisper",
                    "USE_HERMENEIA": "true",
                    "HERMENEIA_MODEL": HERMENEIA_MODELS[body.engine],
                }
        write_env_values(key_file, values)
        settings.caption_engine = values["CAPTION_ENGINE"]
        if "USE_HERMENEIA" in values:
            settings.use_hermeneia = values["USE_HERMENEIA"] == "true"
        if "HERMENEIA_MODEL" in values:
            settings.hermeneia_model = values["HERMENEIA_MODEL"]
        await bus.publish(CaptionEngineChangedEvent())
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

    @app.put("/whole-sentences")
    async def put_whole_sentences(body: WholeSentencesBody) -> dict[str, object]:
        # Kept in .env so the booth's choice survives a restart.
        write_env_values(key_file, {"WHOLE_SENTENCES": "true" if body.whole_sentences else "false"})
        settings.whole_sentences = body.whole_sentences
        state.whole_sentences = body.whole_sentences
        await bus.publish(WholeSentencesEvent(whole_sentences=body.whole_sentences))
        return public_state()

    @app.put("/one-sentence")
    async def put_one_sentence(body: OneSentenceBody) -> dict[str, object]:
        # Kept in .env so the booth's choice survives a restart.
        write_env_values(key_file, {"ONE_SENTENCE": "true" if body.one_sentence else "false"})
        settings.one_sentence = body.one_sentence
        state.one_sentence = body.one_sentence
        await bus.publish(OneSentenceEvent(one_sentence=body.one_sentence))
        return public_state()

    @app.put("/clean-up-wording")
    async def put_clean_up_wording(body: CleanUpWordingBody) -> dict[str, object]:
        # Kept in .env so the booth's choice survives a restart.
        write_env_values(key_file, {"CLEAN_UP_WORDING": "true" if body.clean_up_wording else "false"})
        settings.clean_up_wording = body.clean_up_wording
        state.clean_up_wording = body.clean_up_wording
        await bus.publish(CleanUpWordingEvent(clean_up_wording=body.clean_up_wording))
        return public_state()

    @app.put("/cleanup-context")
    async def put_cleanup_context(body: CleanupContextBody) -> dict[str, object]:
        text = body.text.strip()
        settings.cleanup_context = text
        await bus.publish(CleanupContextEvent(text=text))
        return public_state()

    @app.put("/korean-style")
    async def put_korean_style(body: KoreanStyleBody) -> dict[str, object]:
        # Saved even when empty: an empty file keeps the rule off after a restart.
        text = body.text.strip()
        style_file.parent.mkdir(parents=True, exist_ok=True)
        style_file.write_text(text, encoding="utf-8")
        settings.korean_style = text
        await bus.publish(KoreanStyleEvent(text=text))
        return public_state()

    @app.delete("/korean-style")
    async def reset_korean_style() -> dict[str, object]:
        style_file.unlink(missing_ok=True)
        settings.korean_style = KOREAN_STYLE_DEFAULT
        await bus.publish(KoreanStyleEvent(text=KOREAN_STYLE_DEFAULT))
        return public_state()

    @app.put("/glossary")
    async def put_glossary(body: GlossaryBody) -> dict[str, object]:
        # Saved even when empty: an empty file keeps the glossary off after a restart.
        text = body.text.strip()
        glossary_file.parent.mkdir(parents=True, exist_ok=True)
        glossary_file.write_text(text, encoding="utf-8")
        settings.glossary = text
        await bus.publish(GlossaryEvent(text=text))
        return public_state()

    @app.delete("/glossary")
    async def reset_glossary() -> dict[str, object]:
        glossary_file.unlink(missing_ok=True)
        settings.glossary = GLOSSARY_DEFAULT
        await bus.publish(GlossaryEvent(text=GLOSSARY_DEFAULT))
        return public_state()

    @app.put("/save-transcript")
    async def put_save_transcript(body: SaveTranscriptBody) -> dict[str, object]:
        # Kept in .env so the booth's choice survives a restart.
        write_env_values(key_file, {"SAVE_TRANSCRIPT": "true" if body.save_transcript else "false"})
        settings.save_transcript = body.save_transcript
        state.save_transcript = body.save_transcript
        await bus.publish(SaveTranscriptEvent(save_transcript=body.save_transcript))
        return public_state()

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
