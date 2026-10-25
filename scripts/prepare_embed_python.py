"""Enable pip for the official embeddable Python unzipped beside this script's interpreter."""

from pathlib import Path
import sys


def enable_site_packages(folder: Path) -> None:
    matches = sorted(folder.glob("python*._pth"))
    if not matches:
        raise SystemExit(f"No python*._pth file in {folder}")
    path = matches[0]
    text = path.read_text(encoding="utf-8")
    text = text.replace("#import site", "import site")
    if "Lib\\site-packages" not in text and "Lib/site-packages" not in text:
        text = text.rstrip() + "\r\nLib\\site-packages\r\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    enable_site_packages(Path(sys.executable).resolve().parent)


if __name__ == "__main__":
    main()
