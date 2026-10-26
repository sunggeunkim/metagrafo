from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from functools import partial
from pathlib import Path

from fastapi import FastAPI

from core.event_bus import EventBus
from core.settings import Settings
from features import (
    broadcast_subtitles,
    capture_audio,
    operator_control,
    translate_speech,
    youtube_captions,
)
from features.capture_audio.chunker import SpeechChunker
from features.capture_audio.youtube import download_media, make_job_dir, youtube_video_id
from features.translate_speech.file_run import run_wav


def create_app(
    bus: EventBus | None = None,
    settings: Settings | None = None,
    *,
    start_capture: bool = False,
    load_whisper: bool = False,
    env_file: Path | None = None,
    gemini_connect_factory: Callable[[str], object] | None = None,
    list_input_devices: Callable[[], list[str]] | None = None,
    log_file: Path | None = None,
) -> FastAPI:
    settings = settings or Settings()
    bus = bus or EventBus()
    capture = capture_audio.register(bus, settings) if start_capture else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        translator = app.state.translator
        youtube = app.state.youtube_captions
        await translator.start()
        await youtube.start()
        if capture is not None:
            await capture.start()
        yield
        if capture is not None:
            await capture.stop()
        await translator.stop()
        await youtube.stop()

    app = FastAPI(title="metagrafo", lifespan=lifespan)
    app.state.bus = bus
    app.state.settings = settings
    if list_input_devices is None:
        from features.capture_audio.aten_mic import list_input_names_now

        list_input_devices = list_input_names_now
    operator_control.register(
        app,
        bus,
        settings,
        env_file=env_file,
        list_input_devices=list_input_devices,
        capture_status=capture.status if capture is not None else None,
        log_file=log_file,
    )
    app.state.translator = translate_speech.register(
        app,
        bus,
        settings,
        load_whisper=load_whisper,
        gemini_connect_factory=gemini_connect_factory,
    )
    broadcast_subtitles.register(app, bus, settings)
    app.state.youtube_captions = youtube_captions.register(app, bus, settings)
    return app


app = create_app()


def _configure_logging() -> None:
    """Uvicorn's default config has no root handler, so slice INFO is silent."""
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s: %(message)s"))
        root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)


def _redirect_missing_stdio(log_path: Path) -> Path | None:
    """pythonw has no stdout/stderr, and uvicorn's log setup calls sys.stdout.isatty().

    Returns the log file when output now goes there, or None when a console has it.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return None
    log = open(log_path, "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log
    return log_path


def create_production_app(log_file: Path | None = None) -> FastAPI:
    _configure_logging()
    return create_app(start_capture=True, load_whisper=True, log_file=log_file)


def print_input_devices() -> None:
    import pyaudio

    from features.capture_audio.aten_mic import enumerate_input_devices
    from features.capture_audio.devices import list_input_names

    pa = pyaudio.PyAudio()
    try:
        for name in list_input_names(enumerate_input_devices(pa)):
            print(name)
    finally:
        pa.terminate()


def _non_negative_minutes(value: str) -> float:
    try:
        minutes = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected minutes as a number") from exc
    if not math.isfinite(minutes) or minutes < 0:
        raise argparse.ArgumentTypeError("minutes must be zero or greater")
    return minutes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="metagrafo")
    parser.add_argument("--list-devices", action="store_true")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--youtube",
        help="YouTube sermon URL (save video+wav under metagrafo_job/<yyyyMMddHHmm>/, then translate)",
    )
    source.add_argument("--file", help="Existing wav (skip download)")
    parser.add_argument("--out", help="Caption text file path")
    parser.add_argument("--vad-silence-ms", type=int, default=None)
    parser.add_argument(
        "--pauses",
        type=int,
        choices=(1, 2),
        default=None,
        help="Pauses before a file caption (1 or 2, default 1). Live uses /control.",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Give Whisper the previous caption on the next phrase (default on). Live uses /control.",
    )
    parser.add_argument(
        "--start",
        type=_non_negative_minutes,
        default=0.0,
        metavar="MINUTES",
        help="Begin translation this many minutes into the wav (default 0)",
    )
    parser.add_argument(
        "--end",
        type=_non_negative_minutes,
        default=None,
        metavar="MINUTES",
        help="Stop translation this many minutes into the wav (default: the end of the file)",
    )
    return parser


def _window_seconds(start_minutes: float, end_minutes: float | None) -> tuple[float, float | None]:
    if (
        not math.isfinite(start_minutes)
        or start_minutes < 0
        or (end_minutes is not None and (not math.isfinite(end_minutes) or end_minutes < 0))
    ):
        raise ValueError("start and end are in minutes and must be zero or greater")
    if end_minutes is not None and end_minutes <= start_minutes:
        raise ValueError("--end must be later than --start")
    end_s = None if end_minutes is None else end_minutes * 60.0
    return start_minutes * 60.0, end_s


def run_offline(
    *,
    youtube: str | None,
    file: str | None,
    out: str | None,
    vad_silence_ms: int | None,
    settings: Settings | None = None,
    transcribe=None,
    is_speech: Callable[[bytes], bool] | None = None,
    download: Callable[[str, Path], Path] | None = None,
    now: datetime | None = None,
    cwd: Path | None = None,
    start_minutes: float = 0.0,
    end_minutes: float | None = None,
    pauses: int | None = None,
    condition_on_previous_text: bool | None = None,
) -> Path:
    start_s, end_s = _window_seconds(start_minutes, end_minutes)
    settings = settings or Settings()
    job_path: Path | None = None
    if youtube:
        video_id = youtube_video_id(youtube)
        job_path = make_job_dir(cwd=cwd, now=now)
        out_path = Path(out) if out else job_path / f"{video_id}.en.txt"
        wav_path: Path | None = None
    elif file:
        wav_path = Path(file)
        out_path = Path(out) if out else wav_path.with_suffix(".en.txt")
    else:
        raise ValueError("need --youtube or --file")
    out_path = out_path.resolve()

    vad_ms = settings.vad_min_silence_ms if vad_silence_ms is None else vad_silence_ms
    configured_pauses = settings.vad_pauses if settings.vad_pauses in (1, 2) else 1
    pause_count = configured_pauses if pauses is None else pauses
    if pause_count not in (1, 2):
        raise ValueError("pauses must be 1 or 2")
    use_previous = (
        settings.condition_on_previous_text
        if condition_on_previous_text is None
        else condition_on_previous_text
    )
    if is_speech is None:
        from features.capture_audio.silero import load_is_speech

        is_speech = load_is_speech()
    chunker = SpeechChunker(
        sample_rate=16000,
        frame_samples=512,
        preroll_ms=settings.vad_preroll_ms,
        min_silence_ms=vad_ms,
        max_utterance_s=settings.vad_max_utterance_s,
        min_speech_ms=settings.vad_min_speech_ms,
        pauses_to_cut=pause_count,
        is_speech=is_speech,
    )
    if transcribe is None and settings.hermeneia_url:
        from features.translate_speech.hermeneia_client import make_hermeneia_transcribe

        transcribe = make_hermeneia_transcribe(
            settings.hermeneia_url,
            settings.hermeneia_token,
            settings.hermeneia_model,
            timeout_s=settings.hermeneia_timeout_s,
        )
    elif transcribe is None:
        from features.translate_speech.device_profile import detect_vram_gb, resolve_profile
        from features.translate_speech.whisper_asr import make_transcribe

        profile = resolve_profile(
            vram_gb=detect_vram_gb(),
            model=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        transcribe = make_transcribe(profile)

    def _run(path: Path) -> None:
        run_wav(
            wav_path=path,
            out_path=out_path,
            transcribe=transcribe,
            chunker=chunker,
            mode=settings.translate_mode,
            min_silence_ms=vad_ms,
            start_s=start_s,
            end_s=end_s,
            condition_on_previous_text=use_previous,
        )

    if youtube:
        assert job_path is not None
        fetch = download if download is not None else download_media
        downloaded = fetch(youtube, job_path)
        _run(downloaded)
    else:
        assert wav_path is not None
        _run(wav_path)
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_devices:
        print_input_devices()
        return
    if args.youtube or args.file:
        if args.end is not None and args.end <= args.start:
            parser.error("--end must be later than --start")
        out = run_offline(
            youtube=args.youtube,
            file=args.file,
            out=args.out,
            vad_silence_ms=args.vad_silence_ms,
            start_minutes=args.start,
            end_minutes=args.end,
            pauses=args.pauses,
            condition_on_previous_text=args.condition_on_previous_text,
        )
        print(out)
        return
    log_file = _redirect_missing_stdio(Path("metagrafo.log"))
    import uvicorn

    uvicorn.run(
        partial(create_production_app, log_file=log_file),
        factory=True,
        host="127.0.0.1",
        port=8000,
    )


if __name__ == "__main__":
    main()
