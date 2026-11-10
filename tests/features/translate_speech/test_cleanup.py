from __future__ import annotations

from types import SimpleNamespace

from features.translate_speech.cleanup import gemini_cleanup


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


async def test_cleanup_asks_the_text_model_with_the_shown_lines_as_context(monkeypatch) -> None:
    built, asked = _record_client(monkeypatch, "Paul says nothing can separate us from the love of God.")
    clean = gemini_cleanup("test-key", model="gemini-test-lite")
    cleaned = await clean(
        "And so Paul, what he says is, nothing can separate us from the love of God.",
        ["Today we read from Romans eight.", "Paul writes to the church in Rome."],
    )
    assert cleaned == "Paul says nothing can separate us from the love of God."
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
        "lightly fix grammar",
        "keep the meaning",
        "Add nothing",
        "Keep scripture references and names as spoken",
    ):
        assert asks in instruction
