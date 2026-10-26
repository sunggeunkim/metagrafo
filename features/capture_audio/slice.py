from __future__ import annotations

import asyncio
import logging
import threading

from core.event_bus import EventBus
from core.settings import Settings
from features.capture_audio.chunker import SpeechChunker
from features.capture_audio.gate import CaptureGate
from features.capture_audio.silero import load_is_speech
from features.operator_control.events import AudioDeviceChangedEvent

logger = logging.getLogger(__name__)


class CaptureSlice:
    def __init__(self, bus: EventBus, settings: Settings) -> None:
        self._bus = bus
        self._settings = settings
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # What the booth sees: the device now open, or why the last open failed.
        self._opened: str | None = None
        self._error: str | None = None
        chunker = SpeechChunker(
            sample_rate=16000,
            frame_samples=512,
            preroll_ms=settings.vad_preroll_ms,
            min_silence_ms=settings.vad_min_silence_ms,
            max_utterance_s=settings.vad_max_utterance_s,
            min_speech_ms=settings.vad_min_speech_ms,
            pauses_to_cut=settings.vad_pauses,
            is_speech=load_is_speech(),
        )
        self.gate = CaptureGate(
            bus=bus,
            chunker=chunker,
            sample_rate=16000,
            stream_program_audio=settings.gemini_live(),
        )

    def status(self) -> dict[str, str | None]:
        return {"capturing": self._opened, "error": self._error}

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._opened = None
        self._error = None
        # Each thread gets its own flag, so a thread that outlives stop()'s join still ends.
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(self._stop,), name="capture-audio", daemon=True
        )
        self._thread.start()

    async def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # Off the loop: the capture thread may be waiting on the loop to take a frame.
            await asyncio.to_thread(self._thread.join, 2.0)
            self._thread = None

    async def on_device_changed(self, _event: AudioDeviceChangedEvent) -> None:
        """Reopen capture so _run picks up the device name now on Settings."""
        if self._thread is None:
            return
        await self.stop()
        await self.start()

    def _run(self, stop: threading.Event) -> None:
        try:
            import pyaudio

            from features.capture_audio.aten_mic import (
                PORTAUDIO_LOCK,
                frames_from_stream,
                open_pgm_stream,
            )
        except ImportError:
            logger.exception("PyAudio is not installed; capture is idle")
            return

        with PORTAUDIO_LOCK:
            pa = pyaudio.PyAudio()
        stream = None
        try:
            with PORTAUDIO_LOCK:
                stream, native_rate, channels, chosen = open_pgm_stream(
                    pa,
                    device_name=self._settings.audio_device_name,
                    device_index=self._settings.audio_device_index,
                )
            logger.info("capturing from %s (index %s)", chosen.name, chosen.index)
            if not stop.is_set():
                self._opened = chosen.name
            assert self._loop is not None
            for frame in frames_from_stream(stream, native_rate=native_rate, channels=channels):
                if stop.is_set():
                    break
                fut = asyncio.run_coroutine_threadsafe(self.gate.accept_frame(frame), self._loop)
                try:
                    fut.result(timeout=1.0)
                except Exception:
                    logger.exception("failed to accept capture frame")
        except Exception as exc:
            logger.exception("audio capture stopped")
            # A thread already told to stop must not overwrite the one that replaced it.
            if not stop.is_set():
                self._opened = None
                self._error = str(exc) or type(exc).__name__
        finally:
            with PORTAUDIO_LOCK:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
                pa.terminate()


def register(bus: EventBus, settings: Settings) -> CaptureSlice:
    capture = CaptureSlice(bus, settings)
    bus.subscribe(AudioDeviceChangedEvent, capture.on_device_changed)
    return capture
