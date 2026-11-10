"""Start Metagrafo without a window and tell Start Metagrafo.cmd whether it came up.

Exits 0 once /health answers, 1 when the server stopped or never answered.
Standard library only, so it runs even when Metagrafo's packages are broken.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

HEALTH_URL = "http://127.0.0.1:8000/health"


class Server(Protocol):
    def exited(self) -> bool: ...

    def output(self) -> str: ...


@dataclass(frozen=True)
class Outcome:
    started: bool
    output: str = ""


def wait_for_start(
    server: Server,
    healthy: Callable[[], bool],
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    give_up_after: float = 180.0,
) -> Outcome:
    """Wait until /health answers, the server exits, or it stays silent too long.

    A live process means a slow start, so only an exit ends the wait early.
    """
    deadline = clock() + give_up_after
    while not healthy():
        if server.exited():
            return Outcome(started=False, output=server.output())
        if clock() >= deadline:
            return Outcome(started=False)
        sleep(1.0)
    return Outcome(started=True)


class _WindowlessServer:
    """`pythonw -m main`, detached so it outlives the launcher window.

    Its output, import errors included, is appended to the log, which main.py
    then shows on booth control.
    """

    def __init__(self, root: Path, log_path: Path) -> None:
        self._log_path = log_path
        env = dict(os.environ, PYTHONIOENCODING="utf-8", METAGRAFO_LOG=str(log_path))
        with log_path.open("ab") as log:
            self._start = log.tell()
            self._process = subprocess.Popen(
                [str(Path(sys.executable).with_name("pythonw.exe")), "-u", "-m", "main"],
                cwd=root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            )

    def exited(self) -> bool:
        return self._process.poll() is not None

    def output(self) -> str:
        with self._log_path.open("rb") as log:
            log.seek(self._start)
            return log.read().decode("utf-8", errors="replace")


def _healthy() -> bool:
    # No proxy: a proxy set for the browser must not answer for localhost.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(HEALTH_URL, timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    server = _WindowlessServer(root, root / "metagrafo.log")
    outcome = wait_for_start(server, _healthy, time.monotonic, time.sleep)
    if outcome.output.strip():
        print(outcome.output.rstrip())
    return 0 if outcome.started else 1


if __name__ == "__main__":
    sys.exit(main())
