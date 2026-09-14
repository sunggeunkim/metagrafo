import pytest

from features.capture_audio.devices import DeviceNotFoundError, InputDevice, resolve_input_device

DEVICES = [
    InputDevice(index=0, name="Microphone Array", max_input_channels=2),
    InputDevice(index=1, name="ATEN USB Audio", max_input_channels=2),
    InputDevice(index=2, name="ATEN_Stream_to_USB", max_input_channels=2),
    InputDevice(index=3, name="Speakers", max_input_channels=0),
]

CABLE_DEVICES = [
    InputDevice(index=2, name="CABLE Output (VB-Audio Virtual", max_input_channels=16),
    InputDevice(index=11, name="CABLE Output (VB-Audio Virtual Cable)", max_input_channels=16),
    InputDevice(index=23, name="CABLE Output (VB-Audio Virtual Cable)", max_input_channels=2),
    InputDevice(index=25, name="CABLE Output (VB-Audio Point)", max_input_channels=16),
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


def test_name_match_prefers_two_channel_cable() -> None:
    chosen = resolve_input_device(
        CABLE_DEVICES, name="VB-Audio Virtual Cable", index=None
    )
    assert chosen.index == 23
    assert chosen.max_input_channels == 2


def test_index_still_selects_multichannel_cable() -> None:
    chosen = resolve_input_device(
        CABLE_DEVICES, name="VB-Audio Virtual Cable", index=2
    )
    assert chosen.index == 2
    assert chosen.max_input_channels == 16


def test_default_settings_audio_device_is_vb_cable() -> None:
    from core.settings import Settings

    settings = Settings()
    assert settings.audio_device_name == "VB-Audio Virtual Cable"
    assert settings.audio_device_index is None


def test_missing_device_lists_available_inputs() -> None:
    with pytest.raises(DeviceNotFoundError) as err:
        resolve_input_device(DEVICES, name="BlackHole", index=None)
    assert "Microphone Array" in err.value.available
    assert "ATEN_Stream_to_USB" in err.value.available
    assert "Speakers" not in err.value.available
