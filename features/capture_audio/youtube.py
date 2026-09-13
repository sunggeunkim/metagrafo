from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
}
_VIDEO_ID = re.compile(r"^[\w-]{11}$")


class YouTubeUrlError(ValueError):
    pass


def youtube_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise YouTubeUrlError(f"Not a YouTube URL: {url!r}")
    host = parsed.netloc.lower()
    if host not in _YOUTUBE_HOSTS:
        raise YouTubeUrlError(f"Not a YouTube URL: {url!r}")
    if host.endswith("youtu.be"):
        vid = parsed.path.strip("/").split("/")[0] if parsed.path.strip("/") else ""
    else:
        vid = (parse_qs(parsed.query).get("v") or [""])[0]
    if not _VIDEO_ID.fullmatch(vid):
        raise YouTubeUrlError(f"Not a YouTube watch/youtu.be URL: {url!r}")
    return vid


def make_job_dir(*, cwd: Path | None = None, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d%H%M")
    path = (cwd or Path.cwd()) / "metagrafo_job" / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_media(url: str, dest_dir: Path) -> Path:
    video_id = youtube_video_id(url)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not on PATH; install ffmpeg to extract YouTube audio")
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed") from exc
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bv*+ba/b",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "keepvideo": True,
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    wav_path = dest_dir / f"{video_id}.wav"
    if not wav_path.exists():
        raise RuntimeError(f"yt-dlp did not write {wav_path}")
    return wav_path
