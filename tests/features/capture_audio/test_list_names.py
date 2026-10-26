from features.capture_audio.devices import InputDevice, list_input_names


def test_list_input_names_skips_outputs() -> None:
    devices = [
        InputDevice(index=0, name="ATEN_Stream_to_USB", max_input_channels=2),
        InputDevice(index=1, name="Speakers", max_input_channels=0),
    ]
    assert list_input_names(devices) == ["ATEN_Stream_to_USB"]


class _FakePyAudio:
    def __init__(self, devices: list[dict]) -> None:
        self._devices = devices
        self.terminated = False

    def get_device_count(self) -> int:
        return len(self._devices)

    def get_device_info_by_index(self, index: int) -> dict:
        return self._devices[index]

    def get_host_api_info_by_index(self, index: int) -> dict:
        return {"name": ["MME", "Windows WASAPI"][index]}

    def terminate(self) -> None:
        self.terminated = True


def test_list_input_names_now_names_each_input_once() -> None:
    from features.capture_audio.aten_mic import list_input_names_now

    pa = _FakePyAudio(
        [
            {"name": "HDMI (Elgato 4K S)", "maxInputChannels": 2, "hostApi": 0},
            {"name": "Speakers", "maxInputChannels": 0, "hostApi": 0},
            {"name": "Realtek Mic", "maxInputChannels": 2, "hostApi": 0},
            {"name": "HDMI (Elgato 4K S)", "maxInputChannels": 2, "hostApi": 1},
        ]
    )
    assert list_input_names_now(lambda: pa) == ["HDMI (Elgato 4K S)", "Realtek Mic"]
    assert pa.terminated
