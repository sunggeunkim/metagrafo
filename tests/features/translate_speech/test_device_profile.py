from features.translate_speech.device_profile import resolve_profile


def test_low_vram_defaults_to_medium_int8() -> None:
    profile = resolve_profile(vram_gb=4.0, platform="win32")
    assert profile.model == "medium"
    assert profile.device == "cuda"
    assert profile.compute_type == "int8"


def test_six_gb_defaults_to_large_v3_int8() -> None:
    profile = resolve_profile(vram_gb=6.0, platform="win32")
    assert profile.model == "large-v3"
    assert profile.device == "cuda"
    assert profile.compute_type == "int8"


def test_twelve_gb_defaults_to_large_v3_float16() -> None:
    profile = resolve_profile(vram_gb=12.0, platform="win32")
    assert profile.model == "large-v3"
    assert profile.device == "cuda"
    assert profile.compute_type == "float16"


def test_darwin_defaults_to_cpu() -> None:
    profile = resolve_profile(vram_gb=None, platform="darwin")
    assert profile.device == "cpu"
    assert profile.compute_type == "int8"


def test_env_overrides_vram_heuristic() -> None:
    profile = resolve_profile(
        vram_gb=12.0,
        platform="win32",
        model="medium",
        device="cpu",
        compute_type="int8",
    )
    assert profile.model == "medium"
    assert profile.device == "cpu"
    assert profile.compute_type == "int8"
