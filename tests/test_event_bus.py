from dataclasses import dataclass

from core.event_bus import EventBus


@dataclass(frozen=True, slots=True)
class SampleEvent:
    value: str


@dataclass(frozen=True, slots=True)
class OtherEvent:
    value: str


async def test_subscriber_receives_published_event() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler(event: SampleEvent) -> None:
        seen.append(event.value)

    bus.subscribe(SampleEvent, handler)
    await bus.publish(SampleEvent("hello"))
    assert seen == ["hello"]


async def test_handler_exception_does_not_cancel_sibling() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def boom(_event: SampleEvent) -> None:
        raise RuntimeError("boom")

    async def ok(event: SampleEvent) -> None:
        seen.append(event.value)

    bus.subscribe(SampleEvent, boom)
    bus.subscribe(SampleEvent, ok)
    await bus.publish(SampleEvent("kept"))
    assert seen == ["kept"]


async def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler(event: SampleEvent) -> None:
        seen.append(event.value)

    unsubscribe = bus.subscribe(SampleEvent, handler)
    unsubscribe()
    await bus.publish(SampleEvent("gone"))
    assert seen == []


async def test_publish_dispatches_by_exact_type() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler(event: SampleEvent) -> None:
        seen.append(event.value)

    bus.subscribe(SampleEvent, handler)
    await bus.publish(OtherEvent("nope"))
    assert seen == []
