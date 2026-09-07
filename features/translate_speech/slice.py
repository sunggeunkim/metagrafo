from __future__ import annotations

from fastapi import FastAPI

from core.event_bus import EventBus
from core.settings import Settings
from features.translate_speech.placeholder import register_placeholder


def register(app: FastAPI | None, bus: EventBus, settings: Settings) -> None:
    register_placeholder(bus, mode=settings.translate_mode)
