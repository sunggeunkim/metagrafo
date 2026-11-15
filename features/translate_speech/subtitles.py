from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from features.translate_speech.file_run import Cue

# A caption stays up until the next one starts, but no longer than this past its speech.
HOLD_AFTER_SPEECH_S = 3.0
MIN_SHOW_S = 1.0
# Malgun Gothic ships with Windows and has the Korean glyphs.
FONT = "Malgun Gothic"
STYLE = "FontSize=22,Outline=2,Shadow=0,MarginV=40"


def _timestamp(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def to_srt(cues: Sequence[Cue]) -> str:
    """SubRip text: each caption from the start of its speech until the next caption, at most
    3 s past its speech and at least 1 s on screen."""
    blocks = []
    for index, cue in enumerate(cues):
        until = cue.end_s + HOLD_AFTER_SPEECH_S
        if index + 1 < len(cues):
            until = min(until, cues[index + 1].start_s)
        until = max(until, cue.start_s + MIN_SHOW_S)
        blocks.append(f"{index + 1}\n{_timestamp(cue.start_s)} --> {_timestamp(until)}\n{cue.text}\n")
    return "\n".join(blocks)


def burn_command(video: Path, srt_name: str, out: Path, *, font: str = FONT) -> list[str]:
    """ffmpeg arguments that draw the captions into the picture. The subtitles filter is given a
    bare file name, run from the srt's folder: a Windows path's drive colon breaks the filter."""
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"subtitles={srt_name}:force_style='FontName={font},{STYLE}'",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        str(out),
    ]


def _run_ffmpeg(command: list[str], *, cwd: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not on PATH; install ffmpeg to put the captions on the video")
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg could not burn the captions: {done.stderr.strip()[-800:]}")


def burn_subtitles(
    video: Path,
    srt: Path,
    out: Path,
    *,
    run: Callable[..., None] = _run_ffmpeg,
) -> Path:
    """Write `out`: `video` with the captions in `srt` drawn at the bottom."""
    srt, out = Path(srt).resolve(), Path(out).resolve()
    run(burn_command(Path(video).resolve(), srt.name, out), cwd=srt.parent)
    return out
