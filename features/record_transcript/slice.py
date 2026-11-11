from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import IO

from fastapi import FastAPI

from core.event_bus import EventBus
from core.settings import Settings
from features.operator_control.events import (
    CaptionsStateEvent,
    CleanUpWordingEvent,
    CleanupContextEvent,
    ModeChangedEvent,
    SaveTranscriptEvent,
    WholeSentencesEvent,
)
from features.translate_speech.events import SpeechHeardEvent, SubtitleEvent


class TranscriptRecorder:
    """Writes each finished caption, with its timing, as one JSON record per line.

    For reading back afterwards, not for viewers: `t_s` is seconds since saving
    started, and `delay_s` is how long after the speech began the line was shown.
    """

    def __init__(self, folder: Path, clock: Callable[[], float] = time.time) -> None:
        self._folder = folder
        self._clock = clock
        self._file: IO[str] | None = None
        self._path: Path | None = None
        self._started_at = 0.0

    @property
    def path(self) -> Path | None:
        """The file being written, or None while saving is off."""
        return self._path if self._file is not None else None

    def begin(self, settings: dict[str, object]) -> None:
        self.end()
        self._started_at = self._clock()
        self._folder.mkdir(parents=True, exist_ok=True)
        name = datetime.fromtimestamp(self._started_at).strftime("%Y-%m-%d_%H%M%S")
        # A start within the same second, or after a restart, gets its own file, never an old one.
        suffix = 1
        while True:
            self._path = self._folder / (f"{name}.jsonl" if suffix == 1 else f"{name}_{suffix}.jsonl")
            try:
                # Line-buffered, so a file read mid-service already holds every line shown.
                self._file = self._path.open("x", encoding="utf-8", buffering=1)
                break
            except FileExistsError:
                suffix += 1
        self._write(self._started_at, {"kind": "start", **settings})

    def end(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def caption(self, event: SubtitleEvent) -> None:
        if self._file is None or event.provisional or not event.text.strip():
            return
        record: dict[str, object] = {"kind": "shown", "text": event.text}
        if event.spoken_at is not None:
            record["delay_s"] = round(event.created_at - event.spoken_at, 1)
        record["mode"] = event.mode
        if event.before_cleanup is not None:
            record["before_cleanup"] = event.before_cleanup
        self._write(event.created_at, record)

    def setting(self, name: str, value: object) -> None:
        """A booth change, so later lines can be read against the settings they were made with."""
        if self._file is None:
            return
        self._write(self._clock(), {"kind": "setting", name: value})

    def heard(self, event: SpeechHeardEvent) -> None:
        if self._file is None:
            return
        self._write(event.heard_at, {"kind": "heard", "text": event.text, "language": event.language})

    def _write(self, at: float, record: dict[str, object]) -> None:
        assert self._file is not None
        head = {
            "t_s": round(at - self._started_at, 1),
            "at": datetime.fromtimestamp(at).isoformat(timespec="milliseconds"),
        }
        self._file.write(json.dumps({**head, **record}, ensure_ascii=False) + "\n")


class RecordTranscriptSlice:
    def __init__(
        self,
        recorder: TranscriptRecorder,
        settings: Settings,
        current: Callable[[], dict[str, object]],
    ) -> None:
        self.recorder = recorder
        self._settings = settings
        self._current = current

    async def start(self) -> None:
        if self._settings.save_transcript:
            self.recorder.begin(self._current())

    async def stop(self) -> None:
        self.recorder.end()


def register(
    app: FastAPI,
    bus: EventBus,
    settings: Settings,
    *,
    folder: Path,
    clock: Callable[[], float] | None = None,
) -> RecordTranscriptSlice:
    recorder = TranscriptRecorder(folder, clock or time.time)

    def current() -> dict[str, object]:
        operator = app.state.operator
        return {
            "engine": "gemini_live" if settings.gemini_live() else "whisper",
            "captions_active": operator.captions_active,
            "mode": operator.mode.value,
            "whole_sentences": operator.whole_sentences,
            "clean_up_wording": operator.clean_up_wording,
            "cleanup_context": settings.cleanup_context,
        }

    async def on_caption(event: SubtitleEvent) -> None:
        recorder.caption(event)

    async def on_heard(event: SpeechHeardEvent) -> None:
        recorder.heard(event)

    async def on_save(event: SaveTranscriptEvent) -> None:
        if event.save_transcript:
            recorder.begin(current())
        else:
            recorder.end()

    async def on_captions(event: CaptionsStateEvent) -> None:
        recorder.setting("captions_active", event.is_active)

    async def on_mode(event: ModeChangedEvent) -> None:
        recorder.setting("mode", event.mode)

    async def on_whole_sentences(event: WholeSentencesEvent) -> None:
        recorder.setting("whole_sentences", event.whole_sentences)

    async def on_clean_up_wording(event: CleanUpWordingEvent) -> None:
        recorder.setting("clean_up_wording", event.clean_up_wording)

    async def on_cleanup_context(event: CleanupContextEvent) -> None:
        recorder.setting("cleanup_context", event.text)

    @app.get("/transcript")
    async def get_transcript() -> dict[str, object]:
        path = recorder.path
        return {"saving": path is not None, "file": str(path.resolve()) if path else None}

    bus.subscribe(CaptionsStateEvent, on_captions)
    bus.subscribe(ModeChangedEvent, on_mode)
    bus.subscribe(WholeSentencesEvent, on_whole_sentences)
    bus.subscribe(CleanUpWordingEvent, on_clean_up_wording)
    bus.subscribe(CleanupContextEvent, on_cleanup_context)
    bus.subscribe(SubtitleEvent, on_caption)
    bus.subscribe(SpeechHeardEvent, on_heard)
    bus.subscribe(SaveTranscriptEvent, on_save)
    return RecordTranscriptSlice(recorder, settings, current)
