import pytest

from features.capture_audio.devices import DeviceNotFoundError, InputDevice, resolve_input_device

DEVICES = [
    InputDevice(index=0, name="Microphone Array", max_input_channels=2),
    InputDevice(index=1, name="ATEN USB Audio", max_input_channels=2),
    InputDevice(index=2, name="ATEN_Stream_to_USB", max_input_channels=2),
    InputDevice(index=3, name="Speakers", max_input_channels=0),
]


def test_default_match_prefers_aten_stream_to_usb() -> None:
    chosen = resolve_input_device(DEVICES, name="ATEN_Stream_to_USB", index=None)
    assert chosen.index == 2
    assert chosen.name == "ATEN_Stream_to_USB"


def test_name_override_selects_other_aten_device() -> None:
    chosen = resolve_input_device(DEVICES, name="ATEN USB Audio", index=None)
    assert chosen.name == "ATEN USB Audio"


def test_index_override_wins() -> None:
    chosen = resolve_input_device(DEVICES, name="ATEN_Stream_to_USB", index=0)
    assert chosen.name == "Microphone Array"


def test_missing_device_lists_available_inputs() -> None:
    with pytest.raises(DeviceNotFoundError) as err:
        resolve_input_device(DEVICES, name="BlackHole", index=None)
    assert "Microphone Array" in err.value.available
    assert "ATEN_Stream_to_USB" in err.value.available
    assert "Speakers" not in err.value.available
