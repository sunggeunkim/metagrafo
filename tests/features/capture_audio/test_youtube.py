from datetime import datetime
from pathlib import Path

import pytest

from features.capture_audio.youtube import (
    YouTubeUrlError,
    download_media,
    make_job_dir,
    youtube_video_id,
)


def test_watch_url_extracts_id() -> None:
    assert (
        youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=12s")
        == "dQw4w9WgXcQ"
    )


def test_youtu_be_url_extracts_id() -> None:
    assert youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_rejects_non_youtube_url() -> None:
    with pytest.raises(YouTubeUrlError):
        youtube_video_id("https://example.com/watch?v=dQw4w9WgXcQ")


def test_download_media_rejects_non_youtube_without_writing(tmp_path: Path) -> None:
    dest = tmp_path / "job"
    dest.mkdir()
    with pytest.raises(YouTubeUrlError):
        download_media("https://example.com/watch?v=dQw4w9WgXcQ", dest)
    assert list(dest.iterdir()) == []


def test_job_dir_is_metagrafo_job_yyyyMMddHHmm(tmp_path: Path) -> None:
    path = make_job_dir(cwd=tmp_path, now=datetime(2026, 9, 13, 14, 5))
    assert path == tmp_path / "metagrafo_job" / "202609131405"
    assert path.is_dir()
