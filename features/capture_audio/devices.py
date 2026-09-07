from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InputDevice:
    index: int
    name: str
    max_input_channels: int


class DeviceNotFoundError(LookupError):
    def __init__(self, message: str, available: list[str]) -> None:
        super().__init__(message)
        self.available = available


def _inputs(devices: list[InputDevice]) -> list[InputDevice]:
    return [d for d in devices if d.max_input_channels > 0]


def resolve_input_device(
    devices: list[InputDevice],
    *,
    name: str | None,
    index: int | None,
) -> InputDevice:
    inputs = _inputs(devices)
    available = [d.name for d in inputs]

    if index is not None:
        for device in inputs:
            if device.index == index:
                return device
        raise DeviceNotFoundError(
            f"No input device at index {index}. Available: {available}",
            available,
        )

    needle = (name or "ATEN_Stream_to_USB").casefold()
    matches = [d for d in inputs if needle in d.name.casefold()]
    if not matches:
        raise DeviceNotFoundError(
            f"No input device matching {name!r}. Available: {available}",
            available,
        )
    preferred = [d for d in matches if "aten_stream_to_usb" in d.name.casefold()]
    if preferred:
        return preferred[0]
    return matches[0]


def list_input_names(devices: list[InputDevice]) -> list[str]:
    return [d.name for d in _inputs(devices)]
