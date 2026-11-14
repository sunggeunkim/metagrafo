from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from core.event_bus import EventBus
from core.settings import GLOSSARY_DEFAULT, KOREAN_STYLE_DEFAULT, Settings
from features.translate_speech.events import SubtitleEvent
from main import create_app

START = 1_800_000_000.0


class _Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("SAVE_TRANSCRIPT", raising=False)
    path = tmp_path / ".env"
    path.write_text("", encoding="utf-8")
    return path


def _app(env_file, tmp_path, bus, clock, **settings):
    return create_app(
        bus=bus,
        settings=Settings(caption_engine="whisper", gemini_api_key="", _env_file=env_file, **settings),
        env_file=env_file,
        transcripts_dir=tmp_path / "transcripts",
        transcript_clock=clock,
    )


def _caption(text: str, *, created_at: float, spoken_at: float | None, provisional: bool = False):
    return SubtitleEvent(
        chunk_id="c",
        text=text,
        source_language="ko",
        target_language="en",
        mode="ko_to_en",
        created_at=created_at,
        duration_s=0.0,
        provisional=provisional,
        spoken_at=spoken_at,
    )


def _records(tmp_path: Path) -> list[list[dict]]:
    files = sorted((tmp_path / "transcripts").glob("*.jsonl"))
    return [
        [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()] for f in files
    ]


def _without_clock_time(record: dict) -> dict:
    assert "at" in record
    return {k: v for k, v in record.items() if k != "at"}


def test_saving_writes_each_finished_caption_with_its_timing(env_file, tmp_path) -> None:
    bus = EventBus()
    with TestClient(_app(env_file, tmp_path, bus, _Clock())) as client:
        client.put("/save-transcript", json={"save_transcript": True})
        client.portal.call(
            bus.publish, _caption("Let us", created_at=START + 2.0, spoken_at=START + 1.4, provisional=True)
        )
        client.portal.call(
            bus.publish, _caption("Let us pray.", created_at=START + 3.2, spoken_at=START + 1.4)
        )

    [records] = _records(tmp_path)
    assert records[0]["kind"] == "start"
    assert records[0]["t_s"] == 0.0
    assert [_without_clock_time(r) for r in records[1:]] == [
        {"t_s": 3.2, "kind": "shown", "text": "Let us pray.", "delay_s": 1.8, "mode": "ko_to_en"}
    ]


def test_nothing_is_written_while_saving_is_off(env_file, tmp_path) -> None:
    bus = EventBus()
    with TestClient(_app(env_file, tmp_path, bus, _Clock())) as client:
        client.portal.call(
            bus.publish, _caption("Let us pray.", created_at=START + 3.2, spoken_at=START + 1.4)
        )
    assert _records(tmp_path) == []


class _KoreanSession:
    """A Gemini socket that hears one Korean sentence and says it in English, then stays open."""

    def __init__(self, english: tuple[str, ...]) -> None:
        self._english = english

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    async def send_realtime_input(self, **_kwargs: object) -> None:
        return None

    async def receive(self):
        yield SimpleNamespace(
            go_away=None,
            server_content=SimpleNamespace(
                input_transcription=SimpleNamespace(text="오늘은 기도에 대해"),
                output_transcription=None,
                turn_complete=False,
            ),
        )
        for text in self._english:
            yield SimpleNamespace(
                go_away=None,
                server_content=SimpleNamespace(
                    input_transcription=None,
                    output_transcription=SimpleNamespace(text=text),
                    turn_complete=False,
                ),
            )
        await asyncio.Event().wait()


def _gemini_app(env_file, tmp_path, *, english, cleanup=None, **settings):
    def connect(_api_key: str):
        return lambda _mode: _KoreanSession(english)

    # A short sentence shows once the speaker stops; these tests don't wait the real 5 s for that.
    settings.setdefault("whole_sentences_quiet_s", 0.1)
    return create_app(
        settings=Settings(
            caption_engine="gemini_live",
            gemini_api_key="saved-key",
            save_transcript=True,
            _env_file=env_file,
            **settings,
        ),
        env_file=env_file,
        transcripts_dir=tmp_path / "transcripts",
        gemini_connect_factory=connect,
        gemini_cleanup_factory=cleanup,
    )


def _wait_for(client: TestClient, tmp_path: Path, kind: str) -> list[dict]:
    for _ in range(200):
        files = _records(tmp_path)
        if files and any(r["kind"] == kind for r in files[-1]):
            return files[-1]
        client.get("/health")
        time.sleep(0.01)
    raise AssertionError(f"no {kind} record: {_records(tmp_path)}")


def test_saving_writes_the_korean_gemini_heard(env_file, tmp_path) -> None:
    app = _gemini_app(
        env_file, tmp_path, english=("Today, about prayer.",), clean_up_wording=False
    )
    with TestClient(app) as client:
        records = _wait_for(client, tmp_path, "shown")
    assert [(r["kind"], r.get("text")) for r in records] == [
        ("start", None),
        ("heard", "오늘은 기도에 대해"),
        ("shown", "Today, about prayer."),
    ]
    assert records[1]["language"] == "ko"


def _shouting(_api_key: str, _model: str):
    async def clean(line: str, _shown: list[str], _said: str) -> str:
        return line.upper()

    return clean


def test_a_cleaned_caption_keeps_the_wording_from_before_cleanup(env_file, tmp_path) -> None:
    app = _gemini_app(env_file, tmp_path, english=("Today, about prayer.",), cleanup=_shouting)
    with TestClient(app) as client:
        records = _wait_for(client, tmp_path, "shown")
    [shown] = [r for r in records if r["kind"] == "shown"]
    assert shown["text"] == "TODAY, ABOUT PRAYER."
    assert shown["before_cleanup"] == "Today, about prayer."


def test_a_caption_shown_as_heard_has_no_before_cleanup(env_file, tmp_path) -> None:
    app = _gemini_app(
        env_file, tmp_path, english=("Today, about prayer.",), cleanup=_shouting, clean_up_wording=False
    )
    with TestClient(app) as client:
        records = _wait_for(client, tmp_path, "shown")
    [shown] = [r for r in records if r["kind"] == "shown"]
    assert shown["text"] == "Today, about prayer."
    assert "before_cleanup" not in shown


def test_the_file_notes_the_settings_at_the_start_and_each_booth_change(env_file, tmp_path) -> None:
    bus = EventBus()
    clock = _Clock()
    app = _app(env_file, tmp_path, bus, clock, whole_sentences=True, clean_up_wording=False)
    with TestClient(app) as client:
        client.put("/save-transcript", json={"save_transcript": True})
        clock.now = START + 5.0
        client.put("/captions", json={"is_active": False})
        client.put("/mode", json={"mode": "en_to_en"})
        client.put("/whole-sentences", json={"whole_sentences": False})
        client.put("/clean-up-wording", json={"clean_up_wording": True})
        client.put("/cleanup-context", json={"text": "Sermon on Job 1-2."})
        client.put("/korean-style", json={"text": "해요체로 쓰십시오."})
        client.put("/glossary", json={"text": "Nirvana → 열반"})

    [records] = _records(tmp_path)
    assert _without_clock_time(records[0]) == {
        "t_s": 0.0,
        "kind": "start",
        "engine": "whisper_local",
        "captions_active": True,
        "mode": "ko_to_en",
        "whole_sentences": True,
        "clean_up_wording": False,
        "cleanup_context": "",
        "korean_style": KOREAN_STYLE_DEFAULT,
        "glossary": GLOSSARY_DEFAULT,
    }
    assert [_without_clock_time(r) for r in records[1:]] == [
        {"t_s": 5.0, "kind": "setting", "captions_active": False},
        {"t_s": 5.0, "kind": "setting", "mode": "en_to_en"},
        {"t_s": 5.0, "kind": "setting", "whole_sentences": False},
        {"t_s": 5.0, "kind": "setting", "clean_up_wording": True},
        {"t_s": 5.0, "kind": "setting", "cleanup_context": "Sermon on Job 1-2."},
        {"t_s": 5.0, "kind": "setting", "korean_style": "해요체로 쓰십시오."},
        {"t_s": 5.0, "kind": "setting", "glossary": "Nirvana → 열반"},
    ]


def test_switching_saving_off_and_on_again_starts_a_new_file(env_file, tmp_path) -> None:
    bus = EventBus()
    clock = _Clock()
    with TestClient(_app(env_file, tmp_path, bus, clock)) as client:
        client.put("/save-transcript", json={"save_transcript": True})
        client.portal.call(bus.publish, _caption("First.", created_at=START + 1.0, spoken_at=None))
        client.put("/save-transcript", json={"save_transcript": False})
        client.portal.call(bus.publish, _caption("Unsaved.", created_at=START + 2.0, spoken_at=None))
        clock.now = START + 60.0
        client.put("/save-transcript", json={"save_transcript": True})
        client.portal.call(bus.publish, _caption("Second.", created_at=START + 61.0, spoken_at=None))

    files = _records(tmp_path)
    assert [[r.get("text") for r in f if r["kind"] == "shown"] for f in files] == [
        ["First."],
        ["Second."],
    ]


def test_saving_again_within_the_same_second_still_starts_a_new_file(env_file, tmp_path) -> None:
    bus = EventBus()
    with TestClient(_app(env_file, tmp_path, bus, _Clock())) as client:
        client.put("/save-transcript", json={"save_transcript": True})
        client.portal.call(bus.publish, _caption("First.", created_at=START + 0.5, spoken_at=None))
        client.put("/save-transcript", json={"save_transcript": False})
        client.put("/save-transcript", json={"save_transcript": True})
        client.portal.call(bus.publish, _caption("Second.", created_at=START + 0.7, spoken_at=None))

    files = _records(tmp_path)
    assert [[r["kind"] for r in f] for f in files] == [["start", "shown"], ["start", "shown"]]
    assert [[r.get("text") for r in f if r["kind"] == "shown"] for f in files] == [
        ["First."],
        ["Second."],
    ]
