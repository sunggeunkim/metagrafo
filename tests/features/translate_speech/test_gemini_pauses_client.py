from __future__ import annotations

import io
import json
import wave
from types import SimpleNamespace

from core.settings import GLOSSARY_DEFAULT, KOREAN_STYLE_DEFAULT
from features.translate_speech.gemini_pauses_client import make_gemini_pauses_transcribe

PCM = b"\x00\x10" * 16000  # 1 s at 16 kHz


def _record_client(monkeypatch, reply: dict[str, str] | str) -> tuple[dict[str, object], dict[str, object]]:
    """A Gemini model that answers every piece with `reply`, recording how it was built and asked."""
    from google import genai

    text = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
    built: dict[str, object] = {}
    asked: dict[str, object] = {}

    class RecordingClient:
        def __init__(self, **kwargs: object) -> None:
            built.update(kwargs)

            def generate_content(**kw: object) -> SimpleNamespace:
                asked.update(kw)
                return SimpleNamespace(text=text)

            self.models = SimpleNamespace(generate_content=generate_content)

    monkeypatch.setattr(genai, "Client", RecordingClient)
    return built, asked


def _transcribe(**kwargs: object):
    options = {"thinking_budget": 0, "timeout_s": 15.0}
    options.update(kwargs)
    return make_gemini_pauses_transcribe("test-key", "gemini-test-model", **options)


def _audio(asked: dict[str, object]) -> bytes:
    part = asked["contents"][0]
    assert part.inline_data.mime_type == "audio/wav"
    return part.inline_data.data


def _prompt(asked: dict[str, object]) -> str:
    return asked["contents"][1]


def test_the_piece_goes_to_the_model_as_wav_with_no_thinking(monkeypatch) -> None:
    built, asked = _record_client(monkeypatch, {"heard": "Hello.", "caption": "Hello."})
    caption = _transcribe()(PCM, language="en", task="transcribe", initial_prompt="")
    assert caption == "Hello."
    assert built["api_key"] == "test-key"
    assert asked["model"] == "gemini-test-model"
    config = asked["config"]
    assert config.thinking_config.thinking_budget == 0
    assert config.temperature == 0.0
    assert config.response_mime_type == "application/json"
    with wave.open(io.BytesIO(_audio(asked))) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        assert wav.readframes(wav.getnframes()) == PCM


def test_english_guest_to_korean_captions_asks_for_korean_in_the_booth_style(monkeypatch) -> None:
    _built, asked = _record_client(monkeypatch, {"heard": "God is with us.", "caption": "하나님은 우리와 함께하십니다."})
    caption = _transcribe(korean_style=lambda: KOREAN_STYLE_DEFAULT)(
        PCM, language="en", task="translate", initial_prompt="", target_language="ko"
    )
    assert caption == "하나님은 우리와 함께하십니다."
    instruction = asked["config"].system_instruction
    assert "spoken in English" in instruction
    assert "caption in Korean" in instruction
    assert KOREAN_STYLE_DEFAULT in instruction


def test_korean_sermon_to_english_captions_leaves_out_the_korean_style(monkeypatch) -> None:
    _built, asked = _record_client(monkeypatch, {"heard": "하나님은 사랑이십니다.", "caption": "God is love."})
    _transcribe(korean_style=lambda: KOREAN_STYLE_DEFAULT)(PCM, language="ko", task="translate", initial_prompt="")
    instruction = asked["config"].system_instruction
    assert "spoken in Korean" in instruction
    assert "caption in English" in instruction
    assert KOREAN_STYLE_DEFAULT not in instruction


def test_english_transcribe_writes_down_what_was_said(monkeypatch) -> None:
    _built, asked = _record_client(monkeypatch, {"heard": "Amen.", "caption": "Amen."})
    _transcribe()(PCM, language="en", task="transcribe", initial_prompt="")
    instruction = asked["config"].system_instruction
    assert "spoken in English" in instruction
    assert "caption in English" in instruction
    assert "never translate" in instruction


def test_the_glossary_context_and_shown_lines_go_with_each_piece(monkeypatch) -> None:
    _built, asked = _record_client(monkeypatch, {"heard": "", "caption": "Job suffered."})
    _transcribe(glossary=lambda: GLOSSARY_DEFAULT, context=lambda: "Today's passage is Job 1.")(
        PCM, language="en", task="translate", initial_prompt="We read from the book of Job.", target_language="ko"
    )
    instruction = asked["config"].system_instruction
    assert GLOSSARY_DEFAULT in instruction
    assert "Today's passage is Job 1." in instruction
    assert "Shown before:\nWe read from the book of Job." in _prompt(asked)


def test_no_speech_shows_nothing(monkeypatch) -> None:
    _record_client(monkeypatch, {"heard": "", "caption": "  "})
    assert _transcribe()(PCM, language="en", task="transcribe", initial_prompt="") == ""


def test_a_reply_that_is_not_json_shows_nothing(monkeypatch) -> None:
    _record_client(monkeypatch, "Sorry, I can't help with that.")
    assert _transcribe()(PCM, language="en", task="transcribe", initial_prompt="") == ""


def test_the_model_is_told_never_to_invent_words_from_silence_or_music(monkeypatch) -> None:
    # Without this, gemini-4-hydrogen made a sermon line out of 1 s of silence (3 of 3 tries).
    _built, asked = _record_client(monkeypatch, {"heard": "", "caption": ""})
    _transcribe()(PCM, language="en", task="translate", initial_prompt="", target_language="ko")
    instruction = asked["config"].system_instruction
    assert "silence, room noise, music or singing" in instruction
    assert "An empty caption is always better than words nobody said." in instruction
    assert "If no one is clearly speaking in it, reply with both fields empty." in _prompt(asked)


def test_a_busy_or_rate_limited_gemini_is_tried_once_more(monkeypatch) -> None:
    # The library tries once by default, and a 429 or 503 would lose that piece's caption.
    built, _asked = _record_client(monkeypatch, {"heard": "", "caption": ""})
    _transcribe(timeout_s=12.0)
    options = built["http_options"]
    assert options.timeout == 12000
    retry = options.retry_options
    assert retry.attempts == 2
    assert sorted(retry.http_status_codes) == [429, 503]
    assert retry.max_delay <= 1.0


def test_a_timeout_under_10_s_is_raised_to_10_s(monkeypatch) -> None:
    # Gemini refuses the call outright: "Minimum allowed deadline is 10s."
    built, _asked = _record_client(monkeypatch, {"heard": "", "caption": ""})
    _transcribe(timeout_s=4.0)
    assert built["http_options"].timeout == 10000
