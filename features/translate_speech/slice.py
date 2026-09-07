from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI

from core.event_bus import EventBus
from core.settings import Settings
from features.translate_speech.device_profile import detect_vram_gb, resolve_profile
from features.translate_speech.vocabulary import load_vocabulary
from features.translate_speech.worker import TranslateWorker


class TranslateSlice:
    def __init__(self, worker: TranslateWorker) -> None:
        self.worker = worker
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
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
    prompt = load_vocabulary(Path(settings.church_vocabulary_path))
    if transcribe is None and load_whisper:
        from features.translate_speech.whisper_asr import make_transcribe

        transcribe = make_transcribe(profile)
    if transcribe is None:

        def transcribe(_pcm: bytes, **_kwargs: object) -> str:
            return ""

    worker = TranslateWorker(
        bus,
        transcribe,
        mode=settings.translate_mode,
        initial_prompt=prompt,
        queue_size=settings.translate_queue_size,
    )
    if app is not None:
        app.state.whisper_profile = profile
    return TranslateSlice(worker)
