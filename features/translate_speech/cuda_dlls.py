from __future__ import annotations

import os
import sys
from pathlib import Path


def add_nvidia_dll_dirs() -> list[str]:
    """Put pip-installed CUDA 12 DLLs on the Windows loader path.

    CTranslate2 looks for cublas64_12.dll at encode time. The NVIDIA
    driver is not enough; these wheels ship the CUDA 12 runtime.
    """
    roots: list[Path] = []
    try:
        import nvidia

        roots.append(Path(nvidia.__path__[0]))
    except Exception:
        pass
    site = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if site.is_dir():
        roots.append(site)

    added: list[str] = []
    seen: set[str] = set()
    for root in roots:
        for bin_dir in root.glob("*/bin"):
            key = str(bin_dir.resolve())
            if key in seen or not bin_dir.is_dir():
                continue
            seen.add(key)
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(key)
            os.environ["PATH"] = key + os.pathsep + os.environ.get("PATH", "")
            added.append(key)
    return added
