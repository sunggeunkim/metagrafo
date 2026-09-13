from pathlib import Path

README = Path(__file__).resolve().parent.parent.joinpath("README.md").read_text(encoding="utf-8")


def test_readme_covers_run_and_booth_urls() -> None:
    assert "--list-devices" in README
    assert "create_production_app" in README
    assert "http://127.0.0.1:8000/overlay" in README
    assert "1920" in README and "1080" in README
    assert "http://127.0.0.1:8000/control" in README
    assert "default OFF" in README


def test_readme_covers_worship_mute_and_modes() -> None:
    assert "Captions OFF" in README
    assert "Captions ON" in README
    assert "ko_to_en" in README
    assert "en_to_en" in README
    assert "church_vocabulary.txt" in README
    assert "Restart" in README or "restart" in README


def test_readme_covers_whisper_env_and_medium_restart() -> None:
    assert "WHISPER_MODEL" in README
    assert "WHISPER_DEVICE" in README
    assert "WHISPER_COMPUTE_TYPE" in README
    assert "medium" in README
    assert "restart" in README.lower()
    assert "hot-swap" in README.lower() or "hot swap" in README.lower()
    assert "8–12 GB" in README
    assert "discarded" in README.lower()


def test_readme_covers_aten_windows_and_mac_device_name() -> None:
    assert "ATEN_Stream_to_USB" in README
    assert "Windows-only" in README
    assert "AUDIO_DEVICE_NAME" in README


def test_readme_covers_obs_encoder_not_uc9020_rtmp() -> None:
    assert "OBS" in README
    assert "RTMP" in README
    assert "no captions" in README.lower()
