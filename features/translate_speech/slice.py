from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from functools import partial

from fastapi import FastAPI

from core.event_bus import EventBus
from core.settings import Settings
from features.operator_control.events import (
    CaptionEngineChangedEvent,
    GeminiKeyChangedEvent,
    ModeChangedEvent,
)
from features.translate_speech.device_profile import detect_vram_gb, resolve_profile
from features.translate_speech.cleanup import gemini_cleanup
from features.translate_speech.gemini_live import (
    Cleanup,
    Connect,
    GeminiLiveTranslator,
    gemini_connect,
)
from features.translate_speech.worker import TranscribeFn, TranslateWorker

logger = logging.getLogger(__name__)


class TranslateSlice:
    """Runs one engine at a time. Each engine is built once and kept: a stopped one stays
    subscribed to the bus, so a second copy would take audio nobody reads."""

    def __init__(
        self,
        worker: TranslateWorker | None = None,
        live: GeminiLiveTranslator | None = None,
    ) -> None:
        self.worker = worker
        self.live = live
        self._task: asyncio.Task[None] | None = None
        self._running: str | None = None
        # Switches overlap when the booth clicks twice, and stopping Gemini waits for its socket
        # to close: one switch at a time, each applying what Settings say once it runs.
        self._switching = asyncio.Lock()
        self._settings: Settings | None = None
        self._connect_factory: Callable[[str], Connect] = gemini_connect
        self._bus: EventBus | None = None
        self._mode: str | None = None
        # Whisper on this PC once loaded, and how to load it; where the worker's Whisper runs now.
        self.local_transcribe: TranscribeFn | None = None
        self.load_local: Callable[[], TranscribeFn] | None = None
        self._loading = asyncio.Lock()
        self.worker_runs_on: object = None

    @property
    def whisper_local_ready(self) -> bool:
        """Whisper has loaded on this PC, or is installed and can load when picked."""
        if self.local_transcribe is not None:
            return True
        from features.translate_speech import whisper_asr

        return self.load_local is not None and whisper_asr.installed()

    async def load_local_whisper(self) -> str | None:
        """Load Whisper on this PC if it is not loaded yet. None once it runs; else why not."""
        async with self._loading:
            if self.local_transcribe is not None:
                return None
            if not self.whisper_local_ready:
                return "Whisper is not installed on this PC. Pick Whisper or Seamless on Hermeneia."
            assert self.load_local is not None
            try:
                # Loading takes the model off disk onto the GPU: seconds, a first time minutes.
                self.local_transcribe = await asyncio.to_thread(self.load_local)
            except Exception as error:
                logger.exception("Whisper could not load on this PC")
                return f"Whisper could not load on this PC ({error}). The engine is unchanged."
            return None

    def watch_gemini_key(
        self,
        bus: EventBus,
        settings: Settings,
        connect_factory: Callable[[str], Connect],
        cleanup_factory: Callable[[str, str], Cleanup] = gemini_cleanup,
    ) -> None:
        self._bus = bus
        self._settings = settings
        self._connect_factory = connect_factory
        self._cleanup_factory = cleanup_factory
        bus.subscribe(GeminiKeyChangedEvent, self._on_gemini_key)
        bus.subscribe(CaptionEngineChangedEvent, self._on_engine)
        bus.subscribe(ModeChangedEvent, self._on_mode)

    async def _on_mode(self, event: ModeChangedEvent) -> None:
        """A key pasted later must start Gemini in the mode the booth shows, not the .env one."""
        self._mode = event.mode

    async def _on_gemini_key(self, _event: GeminiKeyChangedEvent) -> None:
        async with self._switching:
            self._gemini(new_key=True)
            await self._apply_engine()

    def _gemini(self, *, new_key: bool = False) -> GeminiLiveTranslator | None:
        """The Gemini translator, built on first use; `new_key` hands it the key just saved.
        None while there is no key."""
        settings = self._settings
        assert settings is not None and self._bus is not None
        key = settings.gemini_api_key.strip()
        if not key:
            return self.live
        if self.live is not None and not new_key:
            return self.live
        connect = self._connect_factory(key)
        cleanup = self._cleanup_factory(key, settings.cleanup_model)
        if self.live is None:
            self.live = GeminiLiveTranslator(
                self._bus,
                connect=connect,
                api_key=key,
                mode=self._mode or settings.translate_mode,
                whole_sentences=settings.whole_sentences,
                quiet_s=settings.whole_sentences_quiet_s,
                clean_up_wording=settings.clean_up_wording,
                cleanup_timeout_s=settings.cleanup_timeout_s,
                cleanup=cleanup,
            )
        else:
            self.live.use_key(key, connect, cleanup=cleanup)
        return self.live

    async def _on_engine(self, _event: CaptionEngineChangedEvent) -> None:
        """Booth control picked an engine or saved a Hermeneia token: run what Settings now say."""
        async with self._switching:
            await self._apply_engine()

    async def _apply_engine(self) -> None:
        """Run the engine Settings name now, so the newest click wins. Holds `_switching`."""
        settings = self._settings
        if settings is None or self._bus is None:
            return
        if settings.gemini_live():
            if self._gemini() is not None:
                await self._run("live")
            return
        runs_on = _whisper_place(settings)
        if runs_on == "this PC" and self.local_transcribe is None:
            logger.warning("Whisper is not loaded on this PC: pick Whisper on this PC, or save the Hermeneia token")
            return
        if self.worker is None:
            self.worker = TranslateWorker(
                self._bus,
                self._whisper_transcribe(runs_on),
                mode=self._mode or settings.translate_mode,
                queue_size=settings.translate_queue_size,
                condition_on_previous_text=settings.condition_on_previous_text,
            )
        elif runs_on != self.worker_runs_on:
            self.worker.use_transcribe(self._whisper_transcribe(runs_on))
        self.worker_runs_on = runs_on
        await self._run("worker")

    def _whisper_transcribe(self, runs_on: object) -> TranscribeFn:
        if runs_on == "this PC":
            assert self.local_transcribe is not None
            return self.local_transcribe
        return _hermeneia_transcribe(self._settings)

    async def _run(self, engine: str) -> None:
        """Stop the engine running now, if it is the other one, and start `engine`."""
        if self._running == engine and self._task is not None:
            return
        if self._running == "live" and self.live is not None:
            # Its quiet wait and cleanup calls are not part of run(): they would show later.
            self.live.suspend()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if engine == "live":
            assert self.live is not None
            # Audio queued while it was stopped is from before the switch.
            self.live.suspend()
            self._task = asyncio.create_task(self.live.run(), name="gemini-live")
        else:
            assert self.worker is not None
            # So are the chunks queued while Gemini ran.
            self.worker.drop_waiting()
            self._task = asyncio.create_task(self.worker.run(), name="translate-speech")
        self._running = engine
        for _ in range(10):
            await asyncio.sleep(0)

    async def start(self) -> None:
        if self.live is not None:
            self._task = asyncio.create_task(self.live.run(), name="gemini-live")
            self._running = "live"
            return
        assert self.worker is not None
        self._task = asyncio.create_task(self.worker.run(), name="translate-speech")
        self._running = "worker"

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._running = None


def _whisper_place(settings: Settings) -> object:
    """Where speech goes for these settings: Hermeneia (address, token and model), or this PC."""
    if settings.hermeneia():
        return (
            "hermeneia",
            settings.hermeneia_url.strip(),
            settings.hermeneia_token.strip(),
            settings.hermeneia_model,
        )
    return "this PC"


def _hermeneia_transcribe(settings: Settings | None) -> TranscribeFn:
    from features.translate_speech.hermeneia_client import make_hermeneia_transcribe

    assert settings is not None
    return make_hermeneia_transcribe(
        settings.hermeneia_url.strip(),
        settings.hermeneia_token.strip(),
        settings.hermeneia_model,
        timeout_s=settings.hermeneia_timeout_s,
    )


def register(
    app: FastAPI | None,
    bus: EventBus,
    settings: Settings,
    *,
    transcribe=None,
    load_whisper: bool = False,
    gemini_connect_factory: Callable[[str], Connect] | None = None,
    gemini_cleanup_factory: Callable[[str, str], Cleanup] | None = None,
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
    # The real cleanup reads the booth's context, Korean style and glossary for every line;
    # a test's stand-in needs neither.
    cleanup_factory = gemini_cleanup_factory or partial(
        gemini_cleanup,
        context=lambda: settings.cleanup_context,
        korean_style=lambda: settings.korean_style,
        glossary=lambda: settings.glossary,
    )
    # Whisper on this PC loads only when it runs: at start when picked, else when first picked.
    # A transcriber handed in (tests, the file run) is Whisper on this PC, already loaded.
    load_local: Callable[[], TranscribeFn] | None = None
    if transcribe is None and load_whisper:

        def load_local() -> TranscribeFn:
            from features.translate_speech.whisper_asr import make_transcribe

            return make_transcribe(profile)

    if settings.gemini_live():
        live = GeminiLiveTranslator(
            bus,
            connect=connect_factory(settings.gemini_api_key),
            api_key=settings.gemini_api_key,
            mode=settings.translate_mode,
            whole_sentences=settings.whole_sentences,
            quiet_s=settings.whole_sentences_quiet_s,
            clean_up_wording=settings.clean_up_wording,
            cleanup_timeout_s=settings.cleanup_timeout_s,
            cleanup=cleanup_factory(settings.gemini_api_key, settings.cleanup_model),
        )
        if app is not None:
            app.state.whisper_profile = profile
        slice_ = TranslateSlice(live=live)
        slice_.local_transcribe = transcribe
        slice_.load_local = load_local
        slice_.watch_gemini_key(bus, settings, connect_factory, cleanup_factory)
        return slice_
    local_transcribe = transcribe
    if transcribe is None and settings.hermeneia():
        transcribe = _hermeneia_transcribe(settings)
    elif load_local is not None:
        try:
            transcribe = local_transcribe = load_local()
        except ImportError:
            logger.warning(
                "Whisper is not installed. Save a Gemini key or the Hermeneia token on booth control."
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
    slice_.local_transcribe = local_transcribe
    slice_.load_local = load_local
    slice_.worker_runs_on = "this PC" if local_transcribe is not None else _whisper_place(settings)
    slice_.watch_gemini_key(bus, settings, connect_factory, cleanup_factory)
    return slice_
