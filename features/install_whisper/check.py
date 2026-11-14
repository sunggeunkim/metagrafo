"""Whether Whisper would run on this PC, read before anything is installed.

nvidia-smi comes with the NVIDIA driver, so the check works on a booth that Start
Metagrafo.cmd set up for Gemini only. It is a forecast: only loading Whisper proves it.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Whisper's CUDA 12 libraries need at least these drivers (NVIDIA's CUDA 12.0 release notes).
MIN_DRIVER = {"win32": (527, 41)}
MIN_DRIVER_OTHER = (525, 60)
# Downloads, in GB: faster-whisper with NVIDIA's cuBLAS and cuDNN wheels (1.40 GB on PyPI for
# Windows, November 2026), and each model.
PACKAGES_DOWNLOAD_GB = 1.4
PACKAGES_DISK_GB = 3.0
MODEL_DOWNLOAD_GB = {"large-v3": 3.1, "large-v2": 3.1, "medium": 1.5, "small": 0.5, "base": 0.15, "tiny": 0.08}
# GPU memory Whisper holds while it runs, after faster-whisper's own benchmark (large-v2, beam 5).
VRAM_GB = {("large", "float16"): 4.5, ("large", "int8"): 3.0, ("medium", "float16"): 2.5, ("medium", "int8"): 1.6}


@dataclass(frozen=True, slots=True)
class Gpu:
    name: str
    total_gb: float
    free_gb: float
    driver: str


def read_gpu() -> Gpu | None:
    """The first NVIDIA GPU, as Whisper would use it, or None with no driver or no GPU."""
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        name, total, free, driver = (part.strip() for part in raw.strip().splitlines()[0].split(","))
        return Gpu(name=name, total_gb=round(float(total) / 1024, 1), free_gb=round(float(free) / 1024, 1), driver=driver)
    except Exception:
        logger.info("No NVIDIA GPU found (nvidia-smi did not answer)")
        return None


def disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def driver_ok(driver: str, platform: str = sys.platform) -> bool:
    numbers = re.findall(r"\d+", driver)
    if len(numbers) < 2:
        return False
    # "527.41" is 527 and 41; "527.4" is 527 and 4, an older driver.
    return (int(numbers[0]), int(numbers[1])) >= MIN_DRIVER.get(platform, MIN_DRIVER_OTHER)


def vram_need_gb(model: str, compute_type: str) -> float | None:
    size = "large" if model.startswith("large") else model
    kind = "int8" if compute_type.startswith("int8") else "float16"
    return VRAM_GB.get((size, kind))


def check_this_pc(model: str, compute_type: str) -> dict[str, object]:
    """What booth control shows before installing: the GPU, the model, the download, a verdict."""
    from features.translate_speech import whisper_asr

    gpu = read_gpu()
    download_gb = PACKAGES_DOWNLOAD_GB + MODEL_DOWNLOAD_GB.get(model, 3.1)
    disk_need_gb = PACKAGES_DISK_GB + MODEL_DOWNLOAD_GB.get(model, 3.1)
    # Packages go beside Python, the model to the Hugging Face cache under the home folder.
    disk_gb = min(disk_free_gb(Path(sys.prefix)), disk_free_gb(Path.home()))
    need_gb = vram_need_gb(model, compute_type)
    lines = []
    if gpu is not None:
        lines.append(f"{gpu.name} · {gpu.total_gb:.1f} GB, {gpu.free_gb:.1f} GB free now · driver {gpu.driver}")
    install = (
        f"The install downloads about {download_gb:.1f} GB (Whisper's packages and the {model} model) "
        f"and can take 10–30 minutes. Disk: {disk_gb:.0f} GB free."
    )
    if whisper_asr.installed():
        verdict, can_install = "installed", False
        lines.append("Whisper is installed. Pick Whisper on this PC under Caption engine.")
    elif gpu is None:
        verdict, can_install = "no_gpu", False
        lines.append(
            "No NVIDIA GPU found. Whisper would run on the processor, too slow for live captions. "
            "Use Whisper or Seamless on Hermeneia."
        )
    elif not driver_ok(gpu.driver):
        verdict, can_install = "old_driver", False
        least = ".".join(str(n) for n in MIN_DRIVER.get(sys.platform, MIN_DRIVER_OTHER))
        lines.append(f"Update the NVIDIA driver first: Whisper needs {least} or newer.")
    elif need_gb is not None and need_gb > gpu.total_gb:
        verdict, can_install = "too_small", False
        lines.append(
            f"Whisper's {model} model needs about {need_gb:.1f} GB of GPU memory, more than this GPU has. "
            "Use Whisper or Seamless on Hermeneia."
        )
    elif disk_gb < disk_need_gb:
        verdict, can_install = "no_disk", False
        lines.append(f"Free up disk space first: the install needs about {disk_need_gb:.0f} GB.")
    elif need_gb is not None and need_gb > gpu.free_gb:
        verdict, can_install = "tight", True
        lines.append(
            f"Whisper on this PC would use {model}, which needs about {need_gb:.1f} GB. It fits this GPU, "
            "but not beside what is using it now, such as OBS. If the stream stutters, use Hermeneia instead."
        )
        lines.append(install)
    elif need_gb is not None and gpu.total_gb < 8:
        verdict, can_install = "tight", True
        lines.append(
            f"Whisper on this PC would use {model}. It fits, but tightly beside OBS. "
            "If the stream stutters, use Hermeneia instead."
        )
        lines.append(install)
    else:
        verdict, can_install = "fits", True
        lines.append(f"Whisper on this PC would use {model}, and it should fit beside OBS.")
        lines.append(install)
    return {
        "verdict": verdict,
        "can_install": can_install,
        "installed": verdict == "installed",
        "gpu": asdict(gpu) if gpu is not None else None,
        "model": model,
        "compute_type": compute_type,
        "download_gb": round(download_gb, 1),
        "disk_free_gb": round(disk_gb, 1),
        "message": "\n".join(lines),
    }
