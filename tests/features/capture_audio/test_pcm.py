import numpy as np

from features.capture_audio.pcm import downmix_to_mono, resample_s16le


def test_stereo_downmix_averages_channels() -> None:
    stereo = np.array([1000, 3000], dtype=np.int16).tobytes()
    mono = np.frombuffer(downmix_to_mono(stereo, channels=2), dtype=np.int16)
    assert list(mono) == [2000]


def test_resample_48k_to_16k_is_one_third_as_long() -> None:
    src = np.full(4800, 1000, dtype=np.int16).tobytes()
    dst = np.frombuffer(resample_s16le(src, src_rate=48000, dst_rate=16000), dtype=np.int16)
    assert len(dst) == 1600
    assert int(dst[0]) == 1000
