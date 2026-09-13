from pathlib import Path

from features.translate_speech.cuda_dlls import add_nvidia_dll_dirs


def test_nvidia_dll_dirs_include_cublas_bin() -> None:
    dirs = add_nvidia_dll_dirs()
    if not dirs:
        return
    assert any(Path(path).joinpath("cublas64_12.dll").is_file() for path in dirs)
