import asyncio
import threading

from core.event_bus import EventBus
from core.settings import Settings
from features.capture_audio import slice as capture_slice
from features.operator_control.events import AudioDeviceChangedEvent


async def test_a_new_input_restarts_capture_on_that_device(monkeypatch) -> None:
    opened: list[str] = []
    second = threading.Event()

    def fake_run(self, stop: threading.Event) -> None:
        opened.append(self._settings.audio_device_name)
        if len(opened) == 2:
            second.set()
        stop.wait()

    monkeypatch.setattr(capture_slice, "load_is_speech", lambda: lambda _frame: False)
    monkeypatch.setattr(capture_slice.CaptureSlice, "_run", fake_run)
    bus = EventBus()
    settings = Settings(_env_file=None, audio_device_name="HDMI (Elgato 4K S)")
    capture = capture_slice.register(bus, settings)

    await capture.start()
    settings.audio_device_name = "Realtek Mic"
    await bus.publish(AudioDeviceChangedEvent())
    assert second.wait(timeout=2.0)
    await capture.stop()

    assert opened == ["HDMI (Elgato 4K S)", "Realtek Mic"]


async def test_a_new_input_does_not_start_capture_that_was_not_running(monkeypatch) -> None:
    ran: list[str] = []
    monkeypatch.setattr(capture_slice, "load_is_speech", lambda: lambda _frame: False)
    monkeypatch.setattr(capture_slice.CaptureSlice, "_run", lambda self, stop: ran.append("run"))
    bus = EventBus()
    capture_slice.register(bus, Settings(_env_file=None))

    await bus.publish(AudioDeviceChangedEvent())

    assert ran == []


async def test_status_says_why_an_input_would_not_open(monkeypatch) -> None:
    import sys
    import types

    class FakePyAudio:
        def terminate(self) -> None:
            pass

    def refuse(_pa, **_kwargs):
        raise OSError(-9999, "Unanticipated host error")

    monkeypatch.setitem(sys.modules, "pyaudio", types.SimpleNamespace(PyAudio=FakePyAudio))
    monkeypatch.setattr("features.capture_audio.aten_mic.open_pgm_stream", refuse)
    monkeypatch.setattr(capture_slice, "load_is_speech", lambda: lambda _frame: False)
    capture = capture_slice.register(EventBus(), Settings(_env_file=None))

    await capture.start()
    for _ in range(200):
        if capture.status()["error"]:
            break
        await asyncio.sleep(0.01)
    await capture.stop()

    assert capture.status() == {"capturing": None, "error": "[Errno -9999] Unanticipated host error"}
