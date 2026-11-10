from __future__ import annotations

from types import SimpleNamespace

from features.translate_speech.compare_run import gemini_translate_sentence


def _record_client(monkeypatch, reply: str) -> tuple[dict[str, object], dict[str, object]]:
    from google import genai

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


async def test_c_asks_the_text_model_to_translate_one_korean_sentence_with_the_english_before_it(monkeypatch) -> None:
    built, asked = _record_client(monkeypatch, " Let us begin today with prayer. ")
    translate = gemini_translate_sentence("test-key", model="gemini-test-lite")
    english = await translate(
        "어, 오늘은 음 기도로 시작하겠습니다.",
        ["Good morning, everyone.", "Please open your Bibles to Romans eight."],
    )
    assert english == "Let us begin today with prayer."
    assert built["api_key"] == "test-key"
    options = built["http_options"]
    api_version = options.get("api_version") if isinstance(options, dict) else getattr(options, "api_version", None)
    assert api_version == "v1beta"
    assert asked["model"] == "gemini-test-lite"
    assert asked["contents"] == (
        "Shown before:\n"
        "Good morning, everyone.\n"
        "Please open your Bibles to Romans eight.\n"
        "\n"
        "Translate:\n"
        "어, 오늘은 음 기도로 시작하겠습니다."
    )
    config = asked["config"]
    assert config.temperature == 0.0
    assert config.automatic_function_calling.disable is True
    instruction = config.system_instruction
    for asks in (
        "into English",
        "Ignore Korean fillers (어, 음, 그, 이제, 뭐)",
        "false starts",
        "Keep the meaning",
        "Add nothing",
        "Keep scripture references and names as spoken",
        "never repeat them",
    ):
        assert asks in instruction


async def test_c_sends_only_the_sentence_when_nothing_was_shown_before(monkeypatch) -> None:
    _built, asked = _record_client(monkeypatch, "Good morning.")
    translate = gemini_translate_sentence("test-key", model="gemini-test-lite")
    await translate("안녕하세요.", [])
    assert asked["contents"] == "Translate:\n안녕하세요."
