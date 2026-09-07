from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from core.event_bus import EventBus
from core.settings import Settings
from features import broadcast_subtitles, operator_control


def create_app(
    bus: EventBus | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    settings = settings or Settings()
    bus = bus or EventBus()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(title="metagrafo", lifespan=lifespan)
    app.state.bus = bus
    app.state.settings = settings
    operator_control.register(app, bus, settings)
    broadcast_subtitles.register(app, bus, settings)
    return app


app = create_app()
