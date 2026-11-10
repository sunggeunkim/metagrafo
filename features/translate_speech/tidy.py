from __future__ import annotations

import re

_SENTENCE_END = ".?!"
_BREAK_OFF = "—–-…"
_FILLER = re.compile(r"u+h+|u+m+|e+r+|h+m+", re.IGNORECASE)


def _repeats(before: str, word: str) -> bool:
    """A stutter: the same word again, with nothing between but a break-off dash."""
    stem = before.rstrip(_BREAK_OFF)
    return stem[-1:].isalnum() and stem.lower() == word.strip(",;:.?!").lower()


def tidy(text: str) -> str:
    """Drop pure fillers (uh, um, er, hmm) and stutters ("the the") from a finished caption line."""
    kept: list[str] = []
    capitalize = False
    for word in text.split():
        if _FILLER.fullmatch(word.strip(",;:.?!")):
            end = word[len(word.rstrip(_SENTENCE_END)):]
            if end and kept:
                # "prayer, hmm." still ends the sentence: the full stop moves back.
                kept[-1] = kept[-1].rstrip(",;:") + end
            if not kept or kept[-1][-1] in _SENTENCE_END:
                capitalize = True
            continue
        if capitalize:
            word = word[:1].upper() + word[1:]
            capitalize = False
        if kept and _repeats(kept[-1], word):
            # "We— we" keeps one "We": the later word, written the way the first began.
            if kept[-1][:1].isupper():
                word = word[:1].upper() + word[1:]
            kept[-1] = word
            continue
        kept.append(word)
    return " ".join(kept)
