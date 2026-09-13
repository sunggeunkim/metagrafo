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


def detect_vram_gb() -> float | None:
    global _vram_gb
    if _vram_gb != "unset":
        return _vram_gb  # type: ignore[return-value]
    try:
        import torch

        if not torch.cuda.is_available():
            _vram_gb = None
        else:
            _vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        _vram_gb = None
    return _vram_gb  # type: ignore[return-value]


def resolve_profile(
    *,
    vram_gb: float | None = None,
    platform: str | None = None,
    model: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
) -> WhisperProfile:
    plat = platform or sys.platform
    resolved_device = device
    if resolved_device is None:
        resolved_device = "cpu" if plat == "darwin" else ("cuda" if vram_gb is not None else "cpu")

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
