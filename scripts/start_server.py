"""Start Metagrafo without a window and tell Start Metagrafo.cmd whether it came up."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


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
