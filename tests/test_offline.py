import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from core.settings import Settings
from features.capture_audio.youtube import YouTubeUrlError
from main import build_parser, main, run_offline

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


def _settings() -> Settings:
    return Settings()


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
        settings=_settings(),
        transcribe=lambda _pcm, **_kwargs: "Hello, everyone.",
        is_speech=_is_speech,
        download=download,
        now=stamp,
        cwd=tmp_path,
        burn=False,
    )
    job = tmp_path / "metagrafo_job" / "202609131430"
    assert downloaded == [("https://www.youtube.com/watch?v=dQw4w9WgXcQ", job)]
    assert result.text == out
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
            settings=_settings(),
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
        settings=_settings(),
        transcribe=lambda _pcm, **_kwargs: "Line.",
        is_speech=_is_speech,
    )
    assert result.text == tmp_path / "sermon.en.txt"
    assert result.text.read_text(encoding="utf-8") == "Line.\n"


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
        settings=_settings(),
        transcribe=lambda _pcm, **_kwargs: "Line.",
        is_speech=_is_speech,
        download=download,
        now=stamp,
        cwd=tmp_path,
        burn=False,
    )
    job = tmp_path / "metagrafo_job" / "202609131405"
    assert result.text == job / "dQw4w9WgXcQ.en.txt"
    assert result.text.is_file()
    assert (job / "dQw4w9WgXcQ.wav").is_file()
    assert (job / "dQw4w9WgXcQ.mp4").is_file()


RATE = 16000


def _two_utterance_wav(path: Path) -> None:
    samples = np.zeros(21 * RATE, dtype=np.int16)
    samples[0:RATE] = 4000
    samples[20 * RATE : 21 * RATE] = 16000
    _write_wav(path, samples)


def _which_utterance(pcm: bytes) -> str:
    mean = float(np.abs(np.frombuffer(pcm, dtype=np.int16)).mean())
    return "early" if mean < 6000 else "late"


def _translate_window(
    tmp_path: Path,
    wav: Path,
    *,
    start_minutes: float = 0.0,
    end_minutes: float | None = None,
) -> str:
    out = tmp_path / f"window-{start_minutes}-{end_minutes}.en.txt"
    run_offline(
        youtube=None,
        file=str(wav),
        out=str(out),
        vad_silence_ms=400,
        settings=_settings(),
        transcribe=lambda pcm, **_kwargs: _which_utterance(pcm),
        is_speech=_is_speech,
        start_minutes=start_minutes,
        end_minutes=end_minutes,
    )
    return out.read_text(encoding="utf-8")


def test_omitted_start_and_end_translate_the_whole_file(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _two_utterance_wav(wav)
    assert _translate_window(tmp_path, wav) == "early\nlate\n"


def test_end_minutes_stops_before_later_speech(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _two_utterance_wav(wav)
    assert _translate_window(tmp_path, wav, end_minutes=0.25) == "early\n"


def test_start_minutes_skips_earlier_speech(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _two_utterance_wav(wav)
    assert _translate_window(tmp_path, wav, start_minutes=0.25) == "late\n"


def test_end_before_start_raises_before_download(tmp_path: Path) -> None:
    called: list[str] = []

    def download(url: str, dest: Path) -> Path:
        called.append(url)
        return dest / "nope.wav"

    with pytest.raises(ValueError, match="later than"):
        run_offline(
            youtube="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            file=None,
            out=str(tmp_path / "out.txt"),
            vad_silence_ms=400,
            settings=_settings(),
            transcribe=lambda _pcm, **_kwargs: "no",
            is_speech=_is_speech,
            download=download,
            start_minutes=2,
            end_minutes=1,
        )
    assert called == []


def test_parser_defaults_to_the_whole_file() -> None:
    args = build_parser().parse_args(["--file", "sermon.wav"])
    assert args.start == 0
    assert args.end is None
    assert args.condition_on_previous_text is None


def test_parser_accepts_condition_on_previous_text() -> None:
    on = build_parser().parse_args(["--file", "sermon.wav", "--condition-on-previous-text"])
    off = build_parser().parse_args(["--file", "sermon.wav", "--no-condition-on-previous-text"])
    assert on.condition_on_previous_text is True
    assert off.condition_on_previous_text is False


def test_main_rejects_end_before_start() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--file", "sermon.wav", "--start", "2", "--end", "1"])
    assert exc.value.code == 2


# The booth's engine and mode, the .srt, and the captioned video.

VIDEO_ID = "dQw4w9WgXcQ"


def _fake_youtube(source: Path):
    def download(_url: str, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        wav = dest_dir / f"{VIDEO_ID}.wav"
        wav.write_bytes(source.read_bytes())
        (dest_dir / f"{VIDEO_ID}.webm").write_bytes(b"video")
        return wav

    return download


def _gemini_settings(**values) -> Settings:
    return Settings(_env_file=None, gemini_api_key="gem-key", **values)


@pytest.fixture
def fake_gemini(monkeypatch) -> tuple[list[dict], list[dict]]:
    """Gemini on pauses, faked where the booth builds it: what it was made with, and each call."""
    calls: list[dict] = []
    made: list[dict] = []

    def make(api_key, model, **kwargs):
        made.append({"api_key": api_key, "model": model})

        def transcribe(_pcm, **kw):
            calls.append(kw)
            return "오늘 우리는" if kw.get("target_language") == "ko" else "Today we will."

        return transcribe

    monkeypatch.setattr("features.translate_speech.gemini_pauses_client.make_gemini_pauses_transcribe", make)
    return made, calls


def test_the_booths_engine_and_mode_write_korean_captions_and_an_srt(tmp_path: Path, fake_gemini) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, np.concatenate([SILENCE, SPEECH, SPEECH, SPEECH, SPEECH, SILENCE]))
    result = run_offline(
        youtube=None,
        file=str(wav),
        out=None,
        vad_silence_ms=400,
        settings=_gemini_settings(caption_engine="gemini_pauses", translate_mode="en_to_ko"),
        is_speech=_is_speech,
    )
    made, calls = fake_gemini
    assert made == [{"api_key": "gem-key", "model": "gemini-4-hydrogen"}]
    assert calls[0]["target_language"] == "ko"
    assert result.text == tmp_path / "sermon.ko.txt"
    assert result.text.read_text(encoding="utf-8") == "오늘 우리는\n"
    assert result.srt == tmp_path / "sermon.ko.srt"
    srt = result.srt.read_text(encoding="utf-8")
    # Speech from 0.96 s, less the booth's preroll (VAD_PREROLL_MS, 160 ms).
    assert srt.startswith("1\n00:00:00,800 --> ")
    assert srt.endswith("오늘 우리는\n")
    assert result.video is None


def test_engine_and_mode_flags_override_the_booth(tmp_path: Path, fake_gemini) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, np.concatenate([SPEECH, SPEECH, SPEECH, SPEECH, SILENCE]))
    result = run_offline(
        youtube=None,
        file=str(wav),
        out=None,
        vad_silence_ms=400,
        settings=_gemini_settings(caption_engine="whisper", translate_mode="ko_to_en"),
        is_speech=_is_speech,
        engine="gemini_pauses",
        mode="en_to_en",
    )
    _made, calls = fake_gemini
    assert calls[0]["language"] == "en"
    assert "target_language" not in calls[0]
    assert result.text.read_text(encoding="utf-8") == "Today we will.\n"


def test_gemini_on_pauses_joins_short_pieces_as_the_booth_does(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    short, gap = np.full(FRAME * 25, 8000), np.zeros(FRAME * 16)  # 0.8 s speech, 0.5 s pause
    _write_wav(wav, np.concatenate([short, gap, short, gap]))
    calls: list[int] = []

    def transcribe(pcm: bytes, **_kw) -> str:
        calls.append(len(pcm))
        return "Joined."

    run_offline(
        youtube=None,
        file=str(wav),
        out=None,
        vad_silence_ms=400,
        settings=_gemini_settings(caption_engine="gemini_pauses", translate_mode="en_to_en"),
        transcribe=transcribe,
        is_speech=_is_speech,
    )
    assert len(calls) == 1


def test_gemini_live_is_refused_before_download(tmp_path: Path) -> None:
    called: list[str] = []

    def download(url: str, dest: Path) -> Path:
        called.append(url)
        return dest / "nope.wav"

    with pytest.raises(ValueError, match="--compare"):
        run_offline(
            youtube=f"https://www.youtube.com/watch?v={VIDEO_ID}",
            file=None,
            out=None,
            vad_silence_ms=None,
            settings=_gemini_settings(caption_engine="gemini_live"),
            is_speech=_is_speech,
            download=download,
            cwd=tmp_path,
        )
    assert called == []


def test_gemini_on_pauses_without_a_key_is_refused(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, SPEECH)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        run_offline(
            youtube=None,
            file=str(wav),
            out=None,
            vad_silence_ms=None,
            settings=Settings(_env_file=None, gemini_api_key=""),
            is_speech=_is_speech,
            engine="gemini_pauses",
        )


def test_korean_on_whisper_is_refused_not_turned_into_english(tmp_path: Path) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, SPEECH)
    with pytest.raises(ValueError, match="Korean"):
        run_offline(
            youtube=None,
            file=str(wav),
            out=None,
            vad_silence_ms=None,
            settings=Settings(_env_file=None, translate_mode="en_to_ko"),
            transcribe=lambda _pcm, **_kw: "no",
            is_speech=_is_speech,
        )


def test_youtube_burns_the_captions_into_the_kept_video(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_wav(source, np.concatenate([SPEECH, SILENCE]))
    burned: list[tuple[Path, Path, Path]] = []

    def burner(video: Path, srt: Path, out: Path) -> Path:
        burned.append((video, srt, out))
        out.write_bytes(b"captioned")
        return out

    result = run_offline(
        youtube=f"https://youtu.be/{VIDEO_ID}",
        file=None,
        out=None,
        vad_silence_ms=400,
        settings=Settings(_env_file=None),
        transcribe=lambda _pcm, **_kw: "Hello.",
        is_speech=_is_speech,
        download=_fake_youtube(source),
        now=datetime(2026, 11, 15, 9, 0),
        cwd=tmp_path,
        burner=burner,
    )
    job = tmp_path / "metagrafo_job" / "202611150900"
    assert burned == [(job / f"{VIDEO_ID}.webm", job / f"{VIDEO_ID}.en.srt", job / f"{VIDEO_ID}.en.captioned.mp4")]
    assert result.video == job / f"{VIDEO_ID}.en.captioned.mp4"
    assert result.srt.read_text(encoding="utf-8").endswith("Hello.\n")


def test_no_burn_and_a_plain_wav_leave_the_video_alone(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _write_wav(source, np.concatenate([SPEECH, SILENCE]))
    burned: list[Path] = []

    def burner(video: Path, srt: Path, out: Path) -> Path:
        burned.append(out)
        return out

    common = dict(
        out=None,
        vad_silence_ms=400,
        settings=Settings(_env_file=None),
        transcribe=lambda _pcm, **_kw: "Hello.",
        is_speech=_is_speech,
        burner=burner,
    )
    no_burn = run_offline(
        youtube=f"https://youtu.be/{VIDEO_ID}", file=None, download=_fake_youtube(source), cwd=tmp_path,
        burn=False, **common,
    )
    plain = run_offline(youtube=None, file=str(source), **common)
    assert burned == []
    assert no_burn.video is None and no_burn.srt.is_file()
    assert plain.video is None and plain.srt == tmp_path / "source.en.srt"


def test_a_wav_with_its_video_is_burned_too(tmp_path: Path) -> None:
    source = tmp_path / "sermon.wav"
    _write_wav(source, np.concatenate([SPEECH, SILENCE]))
    video = tmp_path / "sermon.mp4"
    video.write_bytes(b"video")
    burned: list[tuple[Path, Path, Path]] = []

    def burner(v: Path, srt: Path, out: Path) -> Path:
        burned.append((v, srt, out))
        return out

    result = run_offline(
        youtube=None,
        file=str(source),
        out=None,
        vad_silence_ms=400,
        settings=Settings(_env_file=None),
        transcribe=lambda _pcm, **_kw: "Hello.",
        is_speech=_is_speech,
        video=str(video),
        burner=burner,
    )
    assert burned == [(video, tmp_path / "sermon.en.srt", tmp_path / "sermon.en.captioned.mp4")]
    assert result.video == tmp_path / "sermon.en.captioned.mp4"


def test_parser_takes_engine_mode_video_and_no_burn() -> None:
    args = build_parser().parse_args(
        ["--youtube", "https://youtu.be/x", "--engine", "gemini_pauses", "--mode", "en_to_ko", "--no-burn"]
    )
    assert (args.engine, args.mode, args.burn) == ("gemini_pauses", "en_to_ko", False)
    defaults = build_parser().parse_args(["--file", "a.wav", "--video", "a.mp4"])
    assert (defaults.engine, defaults.mode, defaults.burn, defaults.video) == (None, None, True, "a.mp4")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--file", "a.wav", "--engine", "gemini_live"])
