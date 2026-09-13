from __future__ import annotations

from pathlib import Path


def load_vocabulary(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    return " ".join(line.strip() for line in text.splitlines() if line.strip())
