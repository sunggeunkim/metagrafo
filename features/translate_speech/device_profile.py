from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WhisperProfile:
    model: str
    device: str
    compute_type: str
    vram_gb: float | None


_vram_gb: float | None | str = "unset"


def detect_cuda() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def detect_vram_gb() -> float | None:
    global _vram_gb
    if _vram_gb != "unset":
        return _vram_gb  # type: ignore[return-value]
    try:
        import subprocess

        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        mib = float(raw.strip().splitlines()[0])
        _vram_gb = mib / 1024.0
        return _vram_gb
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            _vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return _vram_gb
    except Exception:
        pass
    _vram_gb = None
    return None


def resolve_profile(
    *,
    vram_gb: float | None = None,
    platform: str | None = None,
    model: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    cuda_available: bool | None = None,
) -> WhisperProfile:
    plat = platform or sys.platform
    resolved_device = device
    if resolved_device is None:
        has_cuda = detect_cuda() if cuda_available is None else cuda_available
        if plat == "darwin":
            resolved_device = "cpu"
        elif vram_gb is not None or has_cuda:
            resolved_device = "cuda"
        else:
            resolved_device = "cpu"

    resolved_model = model
    if resolved_model is None:
        if resolved_device == "cuda" and vram_gb is not None and vram_gb < 6:
            resolved_model = "medium"
        else:
            resolved_model = "large-v3"

    resolved_compute = compute_type
    if resolved_compute is None:
        if resolved_device == "cpu":
            resolved_compute = "int8"
        elif vram_gb is not None and vram_gb >= 12:
            resolved_compute = "float16"
        else:
            resolved_compute = "int8"

    return WhisperProfile(
        model=resolved_model,
        device=resolved_device,
        compute_type=resolved_compute,
        vram_gb=vram_gb,
    )
