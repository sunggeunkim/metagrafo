from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI

from core.event_bus import EventBus
from core.settings import Settings
from features.translate_speech.device_profile import detect_vram_gb, resolve_profile
from features.translate_speech.gemini_live import GeminiLiveTranslator, gemini_connect
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
) -> TranslateSlice:
    profile = resolve_profile(
        vram_gb=detect_vram_gb(),
        model=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    if settings.caption_engine == "gemini_live" and not settings.gemini_api_key.strip():
        logger.warning("CAPTION_ENGINE=gemini_live but GEMINI_API_KEY is empty; using Whisper")
    if settings.gemini_live():
        live = GeminiLiveTranslator(
            bus,
            connect=gemini_connect(settings.gemini_api_key),
            api_key=settings.gemini_api_key,
            mode=settings.translate_mode,
        )
        if app is not None:
            app.state.whisper_profile = profile
        return TranslateSlice(live=live)
    if transcribe is None and settings.hermeneia_url:
        from features.translate_speech.hermeneia_client import make_hermeneia_transcribe

        transcribe = make_hermeneia_transcribe(
            settings.hermeneia_url,
            settings.hermeneia_token,
            settings.hermeneia_model,
            timeout_s=settings.hermeneia_timeout_s,
        )
    elif transcribe is None and load_whisper:
        from features.translate_speech.whisper_asr import make_transcribe

        transcribe = make_transcribe(profile)
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
    return TranslateSlice(worker)
