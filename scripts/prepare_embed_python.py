"""Enable pip for the official embeddable Python unzipped beside this script's interpreter,
and let it import Metagrafo from the repo root."""

from pathlib import Path
import os
import sys


def enable_imports(folder: Path, project: Path) -> None:
    matches = sorted(folder.glob("python*._pth"))
    if not matches:
        raise SystemExit(f"No python*._pth file in {folder}")
    path = matches[0]
    text = path.read_text(encoding="utf-8")
    text = text.replace("#import site", "import site")
    if "Lib\\site-packages" not in text and "Lib/site-packages" not in text:
        text = text.rstrip() + "\r\nLib\\site-packages\r\n"
    # A ._pth file leaves the current folder off sys.path, so `-m main` needs the repo root listed.
    root = os.path.relpath(project, folder)
    if root not in text.splitlines():
        text = text.rstrip() + "\r\n" + root + "\r\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    enable_imports(Path(sys.executable).resolve().parent, Path(__file__).resolve().parent.parent)


if __name__ == "__main__":
    main()
