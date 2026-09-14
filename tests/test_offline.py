import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from core.settings import Settings
from features.capture_audio.youtube import YouTubeUrlError
from main import run_offline

FRAME = 512
SPEECH = np.full(FRAME * 16, 8000, dtype=np.int16)
SILENCE = np.zeros(FRAME * 30, dtype=np.int16)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(samples.astype(np.int16).tobytes())


def _is_speech(frame: bytes) -> bool:
    return float(np.abs(np.frombuffer(frame, dtype=np.int16)).mean()) > 1000


def _settings(tmp_path: Path) -> Settings:
    return Settings(church_vocabulary_path=str(tmp_path / "missing_vocab.txt"))


def test_youtube_downloads_then_writes_captions(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_wav(source, np.concatenate([SPEECH, SILENCE, SPEECH]))
    downloaded: list[tuple[str, Path]] = []

    def download(url: str, dest_dir: Path) -> Path:
        downloaded.append((url, dest_dir))
        dest_dir.mkdir(parents=True, exist_ok=True)
        wav = dest_dir / "dQw4w9WgXcQ.wav"
        wav.write_bytes(source.read_bytes())
        (dest_dir / "dQw4w9WgXcQ.mp4").write_bytes(b"video")
        return wav

    out = tmp_path / "captions.en.txt"
    stamp = datetime(2026, 9, 13, 14, 30)
    result = run_offline(
        youtube="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        file=None,
        out=str(out),
        vad_silence_ms=400,
        settings=_settings(tmp_path),
        transcribe=lambda _pcm, **_kwargs: "Hello, everyone.",
        is_speech=_is_speech,
        download=download,
        now=stamp,
        cwd=tmp_path,
    )
    job = tmp_path / "metagrafo_job" / "202609131430"
    assert downloaded == [("https://www.youtube.com/watch?v=dQw4w9WgXcQ", job)]
    assert result == out
    assert out.read_text(encoding="utf-8") == "Hello, everyone.\nHello, everyone.\n"
    assert (job / "dQw4w9WgXcQ.wav").is_file()
    assert (job / "dQw4w9WgXcQ.mp4").is_file()


def test_rejects_non_youtube_before_download(tmp_path: Path) -> None:
    called: list[str] = []

    def download(url: str, dest: Path) -> Path:
        called.append(url)
        return dest / "nope.wav"

    with pytest.raises(YouTubeUrlError):
        run_offline(
            youtube="https://example.com/watch?v=dQw4w9WgXcQ",
            file=None,
            out=str(tmp_path / "out.txt"),
            vad_silence_ms=None,
            settings=_settings(tmp_path),
            transcribe=lambda _pcm, **_kwargs: "no",
            is_speech=_is_speech,
            download=download,
        )
    assert called == []


def test_file_default_out_suffix(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, SPEECH)
    result = run_offline(
        youtube=None,
        file=str(wav),
        out=None,
        vad_silence_ms=400,
        settings=_settings(tmp_path),
        transcribe=lambda _pcm, **_kwargs: "Line.",
        is_speech=_is_speech,
    )
    assert result == tmp_path / "sermon.en.txt"
    assert result.read_text(encoding="utf-8") == "Line.\n"


def test_youtube_keeps_media_under_timestamped_job_dir(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_wav(source, SPEECH)

    def download(_url: str, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        wav = dest_dir / "dQw4w9WgXcQ.wav"
        wav.write_bytes(source.read_bytes())
        (dest_dir / "dQw4w9WgXcQ.mp4").write_bytes(b"video")
        return wav

    stamp = datetime(2026, 9, 13, 14, 5)
    result = run_offline(
        youtube="https://youtu.be/dQw4w9WgXcQ",
        file=None,
        out=None,
        vad_silence_ms=400,
        settings=_settings(tmp_path),
        transcribe=lambda _pcm, **_kwargs: "Line.",
        is_speech=_is_speech,
        download=download,
        now=stamp,
        cwd=tmp_path,
    )
    job = tmp_path / "metagrafo_job" / "202609131405"
    assert result == job / "dQw4w9WgXcQ.en.txt"
    assert result.is_file()
    assert (job / "dQw4w9WgXcQ.wav").is_file()
    assert (job / "dQw4w9WgXcQ.mp4").is_file()
