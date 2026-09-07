from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[type, list[Callable[..., Awaitable[None]]]] = defaultdict(list)

    def subscribe(
        self, event_type: type[T], handler: Callable[[T], Awaitable[None]]
    ) -> Callable[[], None]:
        self._handlers[event_type].append(handler)

        def unsubscribe() -> None:
            handlers = self._handlers[event_type]
            if handler in handlers:
                handlers.remove(handler)

        return unsubscribe

    async def publish(self, event: object) -> None:
        handlers = list(self._handlers.get(type(event), ()))
        if not handlers:
            return
        results = await asyncio.gather(
            *(handler(event) for handler in handlers),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                logger.exception("event handler failed", exc_info=result)
