from __future__ import annotations

import argparse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.event_bus import EventBus
from core.settings import Settings
from features import broadcast_subtitles, capture_audio, operator_control, translate_speech


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


def main() -> None:
    parser = argparse.ArgumentParser(prog="metagrafo")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    if args.list_devices:
        print_input_devices()
        return
    import uvicorn

    uvicorn.run(create_production_app, factory=True, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
