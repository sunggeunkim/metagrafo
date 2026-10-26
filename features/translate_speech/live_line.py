from __future__ import annotations

_SENTENCE_END = ".?!"


class LiveLine:
    """The open caption. Fragments grow it; a finished sentence clears it."""

    def __init__(self) -> None:
        self._open = ""
        self.opened_at: float | None = None

    def extend(self, fragment: str, *, at: float | None = None) -> str:
        piece = " ".join(fragment.split())
        if not piece:
            return self._open
        if not self._open:
            self.opened_at = at
        self._open = piece if not self._open else f"{self._open} {piece}"
        return self._open

    def closes_sentence(self) -> bool:
        return bool(self._open) and self._open[-1] in _SENTENCE_END

    def commit(self) -> str | None:
        text = self._open.strip()
        self._open = ""
        self.opened_at = None
        return text or None
