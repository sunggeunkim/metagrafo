from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from features.capture_audio.youtube import find_video
from features.translate_speech.file_run import Cue
from features.translate_speech.subtitles import burn_command, burn_subtitles, to_srt


def test_each_caption_shows_from_its_speech_until_the_next() -> None:
    srt = to_srt([Cue(1.0, 4.0, "First."), Cue(5.5, 8.0, "Second.")])
    assert srt == (
        "1\n00:00:01,000 --> 00:00:05,500\nFirst.\n\n"
        "2\n00:00:05,500 --> 00:00:11,000\nSecond.\n"
    )


def test_a_caption_stays_3_s_past_its_speech_at_most() -> None:
    srt = to_srt([Cue(0.0, 2.0, "Alone."), Cue(30.0, 31.0, "Later.")])
    assert "00:00:00,000 --> 00:00:05,000\nAlone." in srt


def test_a_caption_shows_for_a_second_at_least() -> None:
    srt = to_srt([Cue(10.0, 10.2, "Quick."), Cue(10.4, 12.0, "Next.")])
    assert "00:00:10,000 --> 00:00:11,000\nQuick." in srt


def test_hours_and_korean_text() -> None:
    srt = to_srt([Cue(3725.25, 3727.0, "오늘 우리는 욥기를 통해")])
    assert srt == "1\n01:02:05,250 --> 01:02:10,000\n오늘 우리는 욥기를 통해\n"


def test_no_captions_is_an_empty_file() -> None:
    assert to_srt([]) == ""


def test_burn_command_names_the_srt_by_file_name_with_a_korean_font(tmp_path: Path) -> None:
    video = tmp_path / "abc.webm"
    out = tmp_path / "abc.ko.captioned.mp4"
    command = burn_command(video, "abc.ko.srt", out)
    assert command[:4] == ["ffmpeg", "-y", "-i", str(video)]
    vf = command[command.index("-vf") + 1]
    assert vf.startswith("subtitles=abc.ko.srt:force_style='FontName=Malgun Gothic,")
    assert command[-1] == str(out)
    assert "libx264" in command and "aac" in command


@pytest.mark.parametrize("name", ["abc.en.srt", "Sunday's sermon, part [1]; eve.en.srt"])
def test_burn_hands_ffmpeg_the_captions_under_a_plain_name(tmp_path: Path, name: str) -> None:
    # The subtitles filter reads ' , ; [ ] : and \ in a file name as syntax.
    srt = tmp_path / name
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi.\n", encoding="utf-8")
    ran: list[tuple[list[str], str]] = []

    def run(command: list[str], *, cwd: Path) -> None:
        ran.append((command, (Path(cwd) / "captions.srt").read_text(encoding="utf-8")))

    out = burn_subtitles(tmp_path / "abc.mp4", srt, tmp_path / "abc.en.captioned.mp4", run=run)
    assert out == tmp_path / "abc.en.captioned.mp4"
    assert ran == [(burn_command(tmp_path / "abc.mp4", "captions.srt", out), srt.read_text(encoding="utf-8"))]


def test_find_video_skips_the_wav_and_the_captions(tmp_path: Path) -> None:
    for name in ("abc12345678.wav", "abc12345678.en.srt", "abc12345678.en.txt", "abc12345678.webm"):
        (tmp_path / name).write_bytes(b"x")
    assert find_video(tmp_path, "abc12345678") == tmp_path / "abc12345678.webm"


def test_find_video_without_a_video_is_none(tmp_path: Path) -> None:
    (tmp_path / "abc12345678.wav").write_bytes(b"x")
    assert find_video(tmp_path, "abc12345678") is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not on PATH")
def test_real_ffmpeg_burns_korean_into_a_video(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=2", "-f", "lavfi",
         "-i", "anullsrc=r=16000:cl=mono", "-t", "2", "-c:v", "libx264", "-c:a", "aac", str(video)],
        check=True,
        capture_output=True,
    )
    srt = tmp_path / "clip.ko.srt"
    srt.write_text(to_srt([Cue(0.0, 1.5, "오늘 우리는")]), encoding="utf-8")
    out = burn_subtitles(video, srt, tmp_path / "clip.ko.captioned.mp4")
    assert out.stat().st_size > 0
