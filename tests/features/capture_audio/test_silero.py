from features.capture_audio.silero import load_is_speech


def test_silero_accepts_pcm_s16le_frame() -> None:
    is_speech = load_is_speech()
    silence = b"\x00\x00" * 512
    assert is_speech(silence) is False
