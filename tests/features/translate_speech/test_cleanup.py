from __future__ import annotations

import json
from types import SimpleNamespace

from features.translate_speech.cleanup import gemini_cleanup
from features.translate_speech.gemini_live import Cleaned


def _record_client(monkeypatch, reply: str) -> tuple[dict[str, object], dict[str, object]]:
    """A text model that answers every line with `reply`; plain text stands for a finished line."""
    from google import genai

    if not reply.startswith("{"):
        reply = json.dumps({"finished": reply, "unfinished": ""}, ensure_ascii=False)
    built: dict[str, object] = {}
    asked: dict[str, object] = {}

    class RecordingClient:
        def __init__(self, **kwargs: object) -> None:
            built.update(kwargs)

            async def generate_content(**kw: object) -> SimpleNamespace:
                asked.update(kw)
                return SimpleNamespace(text=reply)

            self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

    monkeypatch.setattr(genai, "Client", RecordingClient)
    return built, asked


async def test_cleanup_asks_the_text_model_with_the_shown_lines_as_context(monkeypatch) -> None:
    built, asked = _record_client(
        monkeypatch,
        '{"finished": "Paul says nothing can separate us from the love of God.", "unfinished": ""}',
    )
    clean = gemini_cleanup("test-key", model="gemini-test-lite")
    cleaned = await clean(
        "And so Paul, what he says is, nothing can separate us from the love of God.",
        ["Today we read from Romans eight.", "Paul writes to the church in Rome."],
    )
    assert cleaned == Cleaned(finished="Paul says nothing can separate us from the love of God.")
    assert built["api_key"] == "test-key"
    options = built["http_options"]
    api_version = options.get("api_version") if isinstance(options, dict) else getattr(options, "api_version", None)
    # Same API version as the live socket, so google-genai 2.x and 3.x send the same request.
    assert api_version == "v1beta"
    assert asked["model"] == "gemini-test-lite"
    assert asked["contents"] == (
        "Shown before:\n"
        "Today we read from Romans eight.\n"
        "Paul writes to the church in Rome.\n"
        "\n"
        "Clean up:\n"
        "And so Paul, what he says is, nothing can separate us from the love of God."
    )
    config = asked["config"]
    assert config.temperature == 0.0
    # No tools, so no function-calling loop (and no SDK warning about one in the booth log).
    assert config.automatic_function_calling.disable is True
    instruction = config.system_instruction
    for asks in (
        "remove disfluencies and false starts",
        # Natural, fluent wording, not a light touch: captions read as the speaker would say them.
        "reads as natural, fluent speech in its language",
        "smooth awkward wording and word order",
        # Smoothing changes how a line is said, never what it says.
        "never what it says",
        "every idea, name and detail of the line stays in it",
        "nothing is added",
        "Keep scripture references and names as spoken",
    ):
        assert asks in instruction
    assert "lightly fix grammar" not in instruction


async def test_cleanup_reads_the_booth_context_into_its_instructions_on_each_line(monkeypatch) -> None:
    _built, asked = _record_client(monkeypatch, "Cleaned.")
    context = ["Sermon: the problem of evil and suffering. Scripture: Job 1-2."]
    clean = gemini_cleanup("test-key", model="gemini-test-lite", context=lambda: context[0])

    await clean("A line.", [])
    first = asked["config"].system_instruction
    context[0] = "Speaker: Pastor Frank."
    await clean("Another line.", [])
    second = asked["config"].system_instruction

    assert "Sermon: the problem of evil and suffering. Scripture: Job 1-2." in first
    assert "Speaker: Pastor Frank." in second
    assert "Job 1-2" not in second
    # Background only: it must never turn up in a caption.
    assert "never add it to the line" in first
    assert "Keep scripture references and names as spoken" in first


async def test_cleanup_without_context_keeps_the_plain_instructions(monkeypatch) -> None:
    from features.translate_speech.cleanup import INSTRUCTION

    _built, asked = _record_client(monkeypatch, "Cleaned.")
    clean = gemini_cleanup(
        "test-key", model="gemini-test-lite", context=lambda: "  ", korean_style=lambda: "", glossary=lambda: ""
    )
    await clean("오늘의 말씀입니다.", [])
    assert asked["config"].system_instruction == INSTRUCTION


async def test_cleanup_follows_style_notes_in_the_booth_context(monkeypatch) -> None:
    _built, asked = _record_client(monkeypatch, "Cleaned.")
    style = "Style: use 합니다체, never 해요체."
    await gemini_cleanup("test-key", model="gemini-test-lite", context=lambda: style)("A line.", [])
    instruction = asked["config"].system_instruction
    assert style in instruction
    # The booth can now ask for wording and style, not only names and terms.
    assert "Follow any instructions there about wording or style" in instruction
    assert "never add it to the line" in instruction


async def test_korean_lines_are_cleaned_into_formal_sermon_language_with_no_context(monkeypatch) -> None:
    _built, asked = _record_client(monkeypatch, "Cleaned.")
    await gemini_cleanup("test-key", model="gemini-test-lite")("오늘 만나서 반가워요.", [])
    instruction = asked["config"].system_instruction
    for asks in ("합니다체", "never 해요체 or 반말", "하나님께서", "여러분", "never 당신", "before the verb", "trailing fragments", "'지난달에.'"):
        assert asks in instruction


async def test_the_booth_korean_style_replaces_the_default(monkeypatch) -> None:
    _built, asked = _record_client(monkeypatch, "Cleaned.")
    style = "청소년 예배이니 해요체로 친근하게 쓰십시오."
    clean = gemini_cleanup("test-key", model="gemini-test-lite", korean_style=lambda: style)
    await clean("오늘 만나서 반가워요.", [])
    instruction = asked["config"].system_instruction
    assert style in instruction
    assert "never 해요체 or 반말" not in instruction


async def test_an_empty_korean_style_adds_no_korean_rule(monkeypatch) -> None:
    _built, asked = _record_client(monkeypatch, "Cleaned.")
    clean = gemini_cleanup("test-key", model="gemini-test-lite", korean_style=lambda: "  ")
    await clean("오늘 만나서 반가워요.", [])
    instruction = asked["config"].system_instruction
    assert "When the line is Korean" not in instruction
    assert "합니다체" not in instruction


async def test_cleanup_reads_what_the_speaker_said_to_fix_a_mistranslated_word(monkeypatch) -> None:
    _built, asked = _record_client(monkeypatch, "Cleaned.")
    clean = gemini_cleanup("test-key", model="gemini-test-lite")
    await clean(
        "그래서 지난 몇 주 동안 저희는 동성애 문제에 대해 다루고 있었어요.",
        ["새로운 설교 시리즈를 시작했습니다."],
        "And so over the past couple weeks we have been going over the issue of suffering.",
    )
    assert asked["contents"] == (
        "Shown before:\n"
        "새로운 설교 시리즈를 시작했습니다.\n"
        "\n"
        "Said:\n"
        "And so over the past couple weeks we have been going over the issue of suffering.\n"
        "\n"
        "Clean up:\n"
        "그래서 지난 몇 주 동안 저희는 동성애 문제에 대해 다루고 있었어요."
    )
    instruction = asked["config"].system_instruction
    assert 'Under "Said:"' in instruction
    assert "correct words the translation got wrong" in instruction


async def test_the_default_glossary_goes_with_every_line(monkeypatch) -> None:
    from core.settings import GLOSSARY_DEFAULT

    _built, asked = _record_client(monkeypatch, "Cleaned.")
    await gemini_cleanup("test-key", model="gemini-test-lite")("A line.", [])
    instruction = asked["config"].system_instruction
    assert GLOSSARY_DEFAULT in instruction
    for term in ("하나님", "말씀", "성도 여러분", "개역개정"):
        assert term in GLOSSARY_DEFAULT
    # Only vocabulary for any service: one sermon's terms go in its Additional context.
    for term in ("suffering", "nothingness", "공(空)"):
        assert term not in GLOSSARY_DEFAULT


async def test_the_booth_glossary_replaces_the_default_and_empty_means_none(monkeypatch) -> None:
    from core.settings import GLOSSARY_DEFAULT

    _built, asked = _record_client(monkeypatch, "Cleaned.")
    mine = "Nirvana → 열반"
    await gemini_cleanup("test-key", model="gemini-test-lite", glossary=lambda: mine)("A line.", [])
    assert mine in asked["config"].system_instruction
    assert GLOSSARY_DEFAULT not in asked["config"].system_instruction
    await gemini_cleanup("test-key", model="gemini-test-lite", glossary=lambda: " ")("A line.", [])
    assert "Glossary" not in asked["config"].system_instruction


async def test_cleanup_replies_with_the_finished_sentences_and_the_unfinished_words_after_them(
    monkeypatch,
) -> None:
    _built, asked = _record_client(
        monkeypatch,
        '{"finished": "그리고 이렇게 생각해 보십시오.", "unfinished": "마치 개별적인 존재로서의 당신이"}',
    )
    clean = gemini_cleanup("test-key", model="gemini-test-lite")
    cleaned = await clean("그리고 이렇게 생각해보세요. 마치 개별적인 존재로서의 당신이", [])
    assert cleaned == Cleaned(finished="그리고 이렇게 생각해 보십시오.", unfinished="마치 개별적인 존재로서의 당신이")
    config = asked["config"]
    # Gemini's structured output: every reply is this JSON object, so nothing is parsed by guesswork.
    assert config.response_mime_type == "application/json"
    schema = config.response_json_schema
    assert set(schema["properties"]) == {"finished", "unfinished"}
    assert set(schema["required"]) == {"finished", "unfinished"}
    instruction = config.system_instruction
    for asks in (
        'In "unfinished", put the words at the end of the line that do not finish a sentence yet',
        "copied exactly as they are in the line",
        "never finish them with words from",
    ):
        assert asks in instruction


async def test_a_line_that_must_show_whole_asks_the_model_to_hold_nothing(monkeypatch) -> None:
    _built, asked = _record_client(monkeypatch, "Cleaned.")
    clean = gemini_cleanup("test-key", model="gemini-test-lite")
    await clean("예를 들어, 불교에서는 창시자인 부처에 대한 이야기는", [])
    usual = asked["config"].system_instruction
    await clean("예를 들어, 불교에서는 창시자인 부처에 대한 이야기는", [], hold=False)
    whole = asked["config"].system_instruction
    rule = "This line shows now, even if it ends in the middle of a sentence"
    assert rule not in usual
    assert rule in whole
    assert '"unfinished" is always empty' in whole
    # The tail is part of what shows: it must not be dropped from "finished".
    assert "leave nothing out" in whole
    # Made to show a fragment, the model must not invent its ending (seen live: "멀리" → "멀리하셨습니다").
    assert "no word or ending added" in whole
    # The usual split rules are replaced, not contradicted.
    assert 'when none of it is, "finished" is empty' not in whole
    assert 'Leave them out of "finished"' not in whole


async def test_a_reply_that_is_not_json_says_what_the_model_sent(monkeypatch) -> None:
    import pytest

    _record_client(monkeypatch, "{not json at all")
    with pytest.raises(ValueError, match="not JSON: '{not json at all'"):
        await gemini_cleanup("test-key", model="gemini-test-lite")("A line.", [])


async def test_the_default_korean_style_keeps_honorifics_and_address_to_what_the_speaker_said(monkeypatch) -> None:
    _built, asked = _record_client(monkeypatch, "Cleaned.")
    await gemini_cleanup("test-key", model="gemini-test-lite")("그리고 그는 남은 생애를 보냈어요.", [])
    instruction = asked["config"].system_instruction
    # Seen live: the Buddha became "예수님께서", Frank called himself "청소년부 목사님", and
    # "성도 여러분" was added where the speaker addressed no one.
    for asks in (
        "only for God and Jesus",
        "never make God or Jesus the subject of a sentence about anyone else",
        "목사입니다, not 목사님입니다",
        "never add 여러분 or 성도 여러분 where the speaker addresses no one",
        "stays a question",
        "for every ending, even in a fragment",
    ):
        assert asks in instruction


def test_the_default_glossary_keeps_titles_and_address_to_where_they_belong() -> None:
    from core.settings import GLOSSARY_DEFAULT

    assert "(only when the speaker addresses them) → 성도 여러분" in GLOSSARY_DEFAULT
    assert "the speaker's own title" in GLOSSARY_DEFAULT


def test_the_default_korean_style_gives_only_formal_endings_as_examples() -> None:
    from core.settings import KOREAN_STYLE_DEFAULT

    # A question stays a question, in 합니다체: "알겠습니다." flattens it, "아시겠지요?" is 해요체.
    assert "아시겠습니까?" in KOREAN_STYLE_DEFAULT
    for informal in ("아시겠지요", "그렇지요"):
        assert informal not in KOREAN_STYLE_DEFAULT
    # The 해요체 shapes seen in captions, each with what it becomes.
    for shape in ("했어요 → 했습니다", "있어요 → 있습니다", "거예요 → 것입니다", "것이었죠 → 것이었습니다"):
        assert shape in KOREAN_STYLE_DEFAULT


def test_the_default_glossary_gives_the_speakers_own_title_without_nim() -> None:
    from core.settings import GLOSSARY_DEFAULT

    assert "youth pastor (someone else's title) → 목사님, 장로님, 집사님, 청소년부 목사님" in GLOSSARY_DEFAULT
    assert "youth pastor (the speaker's own title) → 목사, 장로, 집사, 청소년부 목사" in GLOSSARY_DEFAULT


async def test_an_english_line_gets_no_korean_style_and_is_told_to_stay_english(monkeypatch) -> None:
    from core.settings import GLOSSARY_DEFAULT

    _built, asked = _record_client(monkeypatch, "Cleaned.")
    clean = gemini_cleanup("test-key", model="gemini-test-lite")
    # Seen on the real model: with the Korean style and glossary in the instruction, English
    # captions came back in Korean, often as the speaker's own Korean from "Said:".
    await clean(
        "And so Paul, what he says is, nothing can separate us from the love of God.",
        [],
        "그래서 바울이 말하는 것은, 그 무엇도 우리를 하나님의 사랑에서 끊을 수 없다는 것입니다.",
    )
    english = asked["config"].system_instruction
    assert "When the line is Korean" not in english
    assert "합니다체" not in english
    assert "The line is English, so the reply is English" in english
    assert "never translate it" in english
    # The glossary still names terms both ways; only its English side applies.
    assert GLOSSARY_DEFAULT in english
    assert "Use only the English side of the glossary" in english

    await clean("오늘 만나서 반가워요.", [])
    korean = asked["config"].system_instruction
    assert "When the line is Korean" in korean
    assert "The line is English" not in korean

    # Mostly English with one Korean word is English; Korean with an English name is Korean.
    await clean("And so we know that the love of 하나님 never fails us.", [])
    assert "The line is English" in asked["config"].system_instruction
    await clean("오늘은 Romans 8장을 함께 읽겠습니다.", [])
    assert "When the line is Korean" in asked["config"].system_instruction
