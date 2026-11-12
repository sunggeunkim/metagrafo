from __future__ import annotations

_SENTENCE_END = ".?!"


class LiveLine:
    """The open caption. Fragments grow it; a finished sentence clears it."""

    def __init__(self) -> None:
        self._open = ""
        self._word_times: list[float | None] = []
        self.opened_at: float | None = None
        # How many of the first words waited from the caption before, so they wait only once.
        self.carried = 0

    @property
    def text(self) -> str:
        return self._open

    @property
    def word_times(self) -> list[float | None]:
        """When each word of the open line arrived, in order."""
        return list(self._word_times)

    def extend(self, fragment: str, *, at: float | None = None) -> str:
        piece = " ".join(fragment.split())
        if not piece:
            return self._open
        if not self._open:
            self.opened_at = at
        self._open = piece if not self._open else f"{self._open} {piece}"
        self._word_times.extend([at] * len(piece.split()))
        return self._open

    def prepend(self, words: str, times: list[float | None]) -> None:
        """Words held back from the caption before start this one, keeping when they arrived."""
        piece = " ".join(words.split())
        if not piece:
            return
        count = len(piece.split())
        times = (list(times) + [None] * count)[:count]
        self._open = piece if not self._open else f"{piece} {self._open}"
        self._word_times = times + self._word_times
        if times[0] is not None or self.opened_at is None:
            self.opened_at = times[0]
        self.carried += count

    def closes_sentence(self) -> bool:
        return bool(self._open) and self._open[-1] in _SENTENCE_END

    def commit(self) -> str | None:
        text = self._open.strip()
        self._open = ""
        self._word_times = []
        self.opened_at = None
        self.carried = 0
        return text or None
