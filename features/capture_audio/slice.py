from __future__ import annotations

import asyncio
import logging
import threading

from core.event_bus import EventBus
from core.settings import Settings
from features.capture_audio.chunker import SpeechChunker
from features.capture_audio.gate import CaptureGate
from features.capture_audio.silero import load_is_speech

logger = logging.getLogger(__name__)


class CaptureSlice:
    def __init__(self, bus: EventBus, settings: Settings) -> None:
        self._bus = bus
        self._settings = settings
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        chunker = SpeechChunker(
            sample_rate=16000,
            frame_samples=512,
            preroll_ms=settings.vad_preroll_ms,
            min_silence_ms=settings.vad_min_silence_ms,
            max_utterance_s=settings.vad_max_utterance_s,
            min_speech_ms=settings.vad_min_speech_ms,
            is_speech=load_is_speech(),
        )
        self.gate = CaptureGate(bus=bus, chunker=chunker, sample_rate=16000)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="capture-audio", daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        try:
            import pyaudio

            from features.capture_audio.aten_mic import frames_from_stream, open_pgm_stream
        except ImportError:
            logger.exception("PyAudio is not installed; capture is idle")
            return

        pa = pyaudio.PyAudio()
        stream = None
        try:
            stream, native_rate, channels, chosen = open_pgm_stream(
                pa,
                device_name=self._settings.audio_device_name,
                device_index=self._settings.audio_device_index,
            )
            logger.info("capturing from %s (index %s)", chosen.name, chosen.index)
            assert self._loop is not None
            for frame in frames_from_stream(stream, native_rate=native_rate, channels=channels):
                if self._stop.is_set():
                    break
                fut = asyncio.run_coroutine_threadsafe(self.gate.accept_frame(frame), self._loop)
                try:
                    fut.result(timeout=1.0)
                except Exception:
                    logger.exception("failed to accept capture frame")
        except Exception:
            logger.exception("audio capture stopped")
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            pa.terminate()


def register(bus: EventBus, settings: Settings) -> CaptureSlice:
    return CaptureSlice(bus, settings)
