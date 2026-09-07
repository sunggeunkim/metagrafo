from features.capture_audio.devices import InputDevice, list_input_names


def test_list_input_names_skips_outputs() -> None:
    devices = [
        InputDevice(index=0, name="ATEN_Stream_to_USB", max_input_channels=2),
        InputDevice(index=1, name="Speakers", max_input_channels=0),
    ]
    assert list_input_names(devices) == ["ATEN_Stream_to_USB"]
