from __future__ import annotations

from collections.abc import Iterator

from features.capture_audio.devices import InputDevice, resolve_input_device
from features.capture_audio.pcm import downmix_to_mono, resample_s16le

TARGET_RATE = 16000
FRAME_SAMPLES = 512


def enumerate_input_devices(pa) -> list[InputDevice]:
    devices: list[InputDevice] = []
    for index in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(index)
        devices.append(
            InputDevice(
                index=index,
                name=str(info.get("name", "")),
                max_input_channels=int(info.get("maxInputChannels", 0)),
            )
        )
    return devices


def open_pgm_stream(pa, *, device_name: str, device_index: int | None):
    devices = enumerate_input_devices(pa)
    chosen = resolve_input_device(devices, name=device_name, index=device_index)
    info = pa.get_device_info_by_index(chosen.index)
    native_rate = int(info.get("defaultSampleRate") or TARGET_RATE)
    channels = max(1, min(int(info.get("maxInputChannels") or 1), 2))
    import pyaudio

    native_frame = max(FRAME_SAMPLES, int(native_rate * FRAME_SAMPLES / TARGET_RATE))
    kwargs: dict = {
        "format": pyaudio.paInt16,
        "channels": channels,
        "rate": native_rate,
        "input": True,
        "input_device_index": chosen.index,
        "frames_per_buffer": native_frame,
    }
    # PortAudio WASAPI defaults to shared mode (OBS can open the same device).
    stream = pa.open(**kwargs)
    return stream, native_rate, channels, chosen


def frames_from_stream(stream, *, native_rate: int, channels: int) -> Iterator[bytes]:
    native_frame = max(FRAME_SAMPLES, int(native_rate * FRAME_SAMPLES / TARGET_RATE))
    while True:
        raw = stream.read(native_frame, exception_on_overflow=False)
        mono = downmix_to_mono(raw, channels)
        pcm = resample_s16le(mono, src_rate=native_rate, dst_rate=TARGET_RATE)
        samples = memoryview(pcm)
        step = FRAME_SAMPLES * 2
        for offset in range(0, len(samples) - step + 1, step):
            yield bytes(samples[offset : offset + step])
