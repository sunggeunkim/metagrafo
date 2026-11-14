"""Install Whisper into the Python Metagrafo runs on, while it runs."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIREMENTS = Path(__file__).resolve().parents[2] / "requirements-whisper.txt"
_SKIP = {"pip", "setuptools", "wheel"}
# Fetches the model to the Hugging Face cache without loading it onto the GPU.
DOWNLOAD = "import sys\nfrom faster_whisper import download_model\nprint(download_model(sys.argv[1]))\n"
# Every step runs as its own process, the one running now kept here, so stopping Metagrafo
# kills it: a thread cannot be stopped, and Python waits for its threads before it exits.
_running: subprocess.Popen[bytes] | None = None
_stopped = threading.Event()
_lock = threading.Lock()


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


def begin() -> None:
    """A new install: forget a stop left from an earlier one."""
    _stopped.clear()


def _run(command: list[str], log: Callable[[str], None], env: dict[str, str] | None = None) -> int:
    """Run `command`, every output line to `log`; its exit code. Blocks: run it in a thread."""
    global _running
    with _lock:
        if _stopped.is_set():
            return -1
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
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
    freeze: list[str] = []
    if _run([python, "-m", "pip", "freeze"], freeze.append) != 0:
        for line in freeze[-5:] or ["pip freeze failed"]:
            log(line)
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as pinned:
        pinned.write(constraints("\n".join(freeze)))
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


def download_model(model: str, log: Callable[[str], None]) -> bool:
    """Fetch the model in a new Python, which sees the packages pip just added. Blocks."""
    # Progress bars redraw one line with \r: the Log would get it only once it ends.
    env = {**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    return _run([pip_python(sys.executable), "-c", DOWNLOAD, model], log, env=env) == 0


def stop() -> None:
    """Kill the step running now, and start no other until the next install begins."""
    with _lock:
        _stopped.set()
        if _running is not None and _running.poll() is None:
            _running.kill()
