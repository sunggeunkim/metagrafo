from __future__ import annotations

import asyncio
import wave
from pathlib import Path

import pytest

import main as cli
from core.settings import Settings


def _write_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * int(16000 * seconds))


class _QuietSocket:
    async def __aenter__(self) -> _QuietSocket:
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def send_realtime_input(self, **_kwargs: object) -> None:
        return None

    async def receive(self):
        await asyncio.Event().wait()
        yield None


async def _no_wait(_sent_s: float) -> None:
    await asyncio.sleep(0)


def _settings(key: str) -> Settings:
    return Settings(
        gemini_api_key=key,
        cleanup_model="gemini-test-lite",
        cleanup_timeout_s=0.05,
        _env_file=None,
    )


def test_compare_flag_runs_the_side_by_side_file_run_on_the_chosen_minutes(tmp_path, monkeypatch, capsys) -> None:
    called: list[dict[str, object]] = []

    def run_compare(**kwargs: object) -> Path:
        called.append(kwargs)
        return tmp_path / "sermon.compare.md"

    monkeypatch.setattr(cli, "run_compare", run_compare)
    cli.main(["--compare", "--file", "sermon.wav", "--start", "1", "--end", "3"])
    assert called == [
        {"youtube": None, "file": "sermon.wav", "out": None, "start_minutes": 1.0, "end_minutes": 3.0}
    ]
    assert capsys.readouterr().out.strip() == str(tmp_path / "sermon.compare.md")


def test_compare_writes_next_to_the_wav_with_the_saved_key_and_cleanup_model(tmp_path) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, 1.0)
    made: list[tuple[str, ...]] = []

    def connect_factory(api_key: str):
        made.append(("connect", api_key))
        return lambda _mode: _QuietSocket()

    def cleanup_factory(api_key: str, model: str):
        made.append(("cleanup", api_key, model))
        return None

    def translate_factory(api_key: str, model: str):
        made.append(("translate", api_key, model))
        return None

    out = asyncio.run(
        cli.run_compare_async(
            youtube=None,
            file=str(wav),
            out=None,
            settings=_settings("saved-key"),
            connect_factory=connect_factory,
            cleanup_factory=cleanup_factory,
            translate_factory=translate_factory,
            pace=_no_wait,
        )
    )
    assert out == (tmp_path / "sermon.compare.md").resolve()
    assert out.read_text(encoding="utf-8").startswith("# sermon.wav: B vs C\n")
    assert made == [
        ("connect", "saved-key"),
        ("cleanup", "saved-key", "gemini-test-lite"),
        ("translate", "saved-key", "gemini-test-lite"),
    ]


def test_compare_needs_a_gemini_key(tmp_path) -> None:
    wav = tmp_path / "sermon.wav"
    _write_wav(wav, 1.0)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        asyncio.run(cli.run_compare_async(youtube=None, file=str(wav), out=None, settings=_settings("  ")))
