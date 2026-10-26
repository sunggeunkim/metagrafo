from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable

from fastapi import FastAPI

from core.event_bus import EventBus
from core.settings import Settings
from features.operator_control.events import GeminiKeyChangedEvent, ModeChangedEvent
from features.translate_speech.device_profile import detect_vram_gb, resolve_profile
from features.translate_speech.gemini_live import Connect, GeminiLiveTranslator, gemini_connect
from features.translate_speech.worker import TranslateWorker

logger = logging.getLogger(__name__)


class TranslateSlice:
    def __init__(
        self,
        worker: TranslateWorker | None = None,
        live: GeminiLiveTranslator | None = None,
    ) -> None:
        self.worker = worker
        self.live = live
        self._task: asyncio.Task[None] | None = None
        self._settings: Settings | None = None
        self._connect_factory: Callable[[str], Connect] = gemini_connect
        self._bus: EventBus | None = None
        self._mode: str | None = None

    def watch_gemini_key(
        self,
        bus: EventBus,
        settings: Settings,
        connect_factory: Callable[[str], Connect],
    ) -> None:
        self._bus = bus
        self._settings = settings
        self._connect_factory = connect_factory
        bus.subscribe(GeminiKeyChangedEvent, self._on_gemini_key)
        bus.subscribe(ModeChangedEvent, self._on_mode)

    async def _on_mode(self, event: ModeChangedEvent) -> None:
        """A key pasted later must start Gemini in the mode the booth shows, not the .env one."""
        self._mode = event.mode

    async def _on_gemini_key(self, _event: GeminiKeyChangedEvent) -> None:
        settings = self._settings
        bus = self._bus
        if settings is None or bus is None:
            return
        key = settings.gemini_api_key.strip()
        if not key:
            return
        connect = self._connect_factory(key)
        if self.live is not None:
            self.live.use_key(key, connect)
            for _ in range(10):
                await asyncio.sleep(0)
            return
        live = GeminiLiveTranslator(
            bus,
            connect=connect,
            api_key=key,
            mode=self._mode or settings.translate_mode,
        )
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self.worker = None
        self.live = live
        self._task = asyncio.create_task(self.live.run(), name="gemini-live")
        for _ in range(10):
            await asyncio.sleep(0)

    async def start(self) -> None:
        if self.live is not None:
            self._task = asyncio.create_task(self.live.run(), name="gemini-live")
            return
        assert self.worker is not None
        self._task = asyncio.create_task(self.worker.run(), name="translate-speech")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None


def register(
    app: FastAPI | None,
    bus: EventBus,
    settings: Settings,
    *,
    transcribe=None,
    load_whisper: bool = False,
    gemini_connect_factory: Callable[[str], Connect] | None = None,
) -> TranslateSlice:
    profile = resolve_profile(
        vram_gb=detect_vram_gb(),
        model=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    if settings.caption_engine == "gemini_live" and not settings.gemini_api_key.strip():
        logger.warning("CAPTION_ENGINE=gemini_live but GEMINI_API_KEY is empty; using Whisper")
    connect_factory = gemini_connect_factory or gemini_connect
    if settings.gemini_live():
        live = GeminiLiveTranslator(
            bus,
            connect=connect_factory(settings.gemini_api_key),
            api_key=settings.gemini_api_key,
            mode=settings.translate_mode,
        )
        if app is not None:
            app.state.whisper_profile = profile
        slice_ = TranslateSlice(live=live)
        slice_.watch_gemini_key(bus, settings, connect_factory)
        return slice_
    if transcribe is None and settings.hermeneia_url:
        from features.translate_speech.hermeneia_client import make_hermeneia_transcribe

        transcribe = make_hermeneia_transcribe(
            settings.hermeneia_url,
            settings.hermeneia_token,
            settings.hermeneia_model,
            timeout_s=settings.hermeneia_timeout_s,
        )
    elif transcribe is None and load_whisper:
        try:
            from features.translate_speech.whisper_asr import make_transcribe

            transcribe = make_transcribe(profile)
        except ImportError:
            logger.warning(
                "Whisper is not installed. Save a Gemini key on booth control."
            )
    if transcribe is None:

        def transcribe(_pcm: bytes, **_kwargs: object) -> str:
            return ""

    worker = TranslateWorker(
        bus,
        transcribe,
        mode=settings.translate_mode,
        queue_size=settings.translate_queue_size,
        condition_on_previous_text=settings.condition_on_previous_text,
    )
    if app is not None:
        app.state.whisper_profile = profile
    slice_ = TranslateSlice(worker)
    slice_.watch_gemini_key(bus, settings, connect_factory)
    return slice_
