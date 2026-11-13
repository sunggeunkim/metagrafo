"""Install Whisper into the Python Metagrafo runs on, while it runs."""

from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIREMENTS = Path(__file__).resolve().parents[2] / "requirements-whisper.txt"
_SKIP = {"pip", "setuptools", "wheel"}
# The pip running now, so stopping Metagrafo can stop it too.
_running: subprocess.Popen[bytes] | None = None


def pip_python(executable: str) -> str:
    """Start Metagrafo.cmd runs pythonw, which has no console for pip to write to."""
    path = Path(executable)
    if path.name.lower() == "pythonw.exe" and path.with_name("python.exe").exists():
        return str(path.with_name("python.exe"))
    return executable


def constraints(freeze: str) -> str:
    """Pin what is installed: Windows locks the files of packages Metagrafo has open, such as
    numpy, so an upgrade would fail halfway."""
    pins = []
    for line in freeze.splitlines():
        name, sep, _version = line.partition("==")
        if sep and name.strip().lower() not in _SKIP:
            pins.append(line.strip())
    return "".join(pin + "\n" for pin in pins)


def _run(command: list[str], log: Callable[[str], None]) -> int:
    global _running
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    _running = process
    try:
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.decode(errors="replace").rstrip()
            if line:
                log(line)
        return process.wait()
    finally:
        _running = None


def run_pip(log: Callable[[str], None]) -> bool:
    """pip install requirements-whisper.txt, every line to `log`. Blocks: run it in a thread."""
    python = pip_python(sys.executable)
    freeze = subprocess.run(
        [python, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if freeze.returncode != 0:
        log(freeze.stderr.strip() or "pip freeze failed")
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as pinned:
        pinned.write(constraints(freeze.stdout))
    try:
        code = _run(
            [
                python, "-m", "pip", "install",
                "--disable-pip-version-check", "--no-warn-script-location",
                "-r", str(REQUIREMENTS), "-c", pinned.name,
            ],
            log,
        )
    finally:
        Path(pinned.name).unlink(missing_ok=True)
    return code == 0


def download_model(model: str) -> None:
    """Fetch the model to the Hugging Face cache without loading it onto the GPU."""
    import importlib

    importlib.invalidate_caches()
    from faster_whisper import download_model as fetch

    fetch(model)


def stop() -> None:
    if _running is not None and _running.poll() is None:
        _running.kill()
