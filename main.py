from __future__ import annotations

import argparse
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI

from core.event_bus import EventBus
from core.settings import Settings
from features import broadcast_subtitles, capture_audio, operator_control, translate_speech
from features.capture_audio.chunker import SpeechChunker
from features.capture_audio.youtube import download_media, make_job_dir, youtube_video_id
from features.translate_speech.file_run import run_wav
from features.translate_speech.vocabulary import load_vocabulary


def create_app(
    bus: EventBus | None = None,
    settings: Settings | None = None,
    *,
    start_capture: bool = False,
    load_whisper: bool = False,
) -> FastAPI:
    settings = settings or Settings()
    bus = bus or EventBus()
    capture = capture_audio.register(bus, settings) if start_capture else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        translator = app.state.translator
        await translator.start()
        if capture is not None:
            await capture.start()
        yield
        if capture is not None:
            await capture.stop()
        await translator.stop()

    app = FastAPI(title="metagrafo", lifespan=lifespan)
    app.state.bus = bus
    app.state.settings = settings
    operator_control.register(app, bus, settings)
    app.state.translator = translate_speech.register(
        app, bus, settings, load_whisper=load_whisper
    )
    broadcast_subtitles.register(app, bus, settings)
    return app


app = create_app()


def create_production_app() -> FastAPI:
    return create_app(start_capture=True, load_whisper=True)


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
    return parser


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
) -> Path:
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
        is_speech=is_speech,
    )
    if transcribe is None:
        from features.translate_speech.device_profile import detect_vram_gb, resolve_profile
        from features.translate_speech.whisper_asr import make_transcribe

        profile = resolve_profile(
            vram_gb=detect_vram_gb(),
            model=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
        transcribe = make_transcribe(profile)
    prompt = load_vocabulary(Path(settings.church_vocabulary_path))

    def _run(path: Path) -> None:
        run_wav(
            wav_path=path,
            out_path=out_path,
            transcribe=transcribe,
            chunker=chunker,
            mode=settings.translate_mode,
            initial_prompt=prompt,
            min_silence_ms=vad_ms,
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
        out = run_offline(
            youtube=args.youtube,
            file=args.file,
            out=args.out,
            vad_silence_ms=args.vad_silence_ms,
        )
        print(out)
        return
    import uvicorn

    uvicorn.run(create_production_app, factory=True, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
