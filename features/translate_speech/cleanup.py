from __future__ import annotations

from collections.abc import Callable

from features.translate_speech.gemini_live import Cleanup

INSTRUCTION = (
    "You clean up live captions of a church sermon. They were machine-translated while the "
    "pastor was still speaking, so they can backtrack or read awkwardly. Rewrite the line under "
    '"Clean up:": remove disfluencies and false starts, lightly fix grammar, and keep the meaning. '
    "Add nothing. Keep scripture references and names as spoken. Keep the line's language. "
    'The lines under "Shown before:" are already on screen; use them only to understand the '
    "line and never repeat them. Reply with the cleaned line only."
)


def prompt_for(line: str, shown: list[str]) -> str:
    before = "".join(f"{text}\n" for text in shown)
    return f"Shown before:\n{before}\nClean up:\n{line}" if shown else f"Clean up:\n{line}"


def instruction_with(context: str) -> str:
    """The instruction, plus what the booth wrote about this service, when it wrote anything."""
    if not context.strip():
        return INSTRUCTION
    return (
        f"{INSTRUCTION}\n\nAbout this service, from the booth. Use it only to get names, "
        f"scripture references and terms right; never add it to the line:\n{context.strip()}"
    )


def gemini_cleanup(api_key: str, model: str, context: Callable[[], str] | None = None) -> Cleanup:
    """A fast Gemini text model (`CLEANUP_MODEL`) that rewrites one finished caption line.

    `context` is read for every line, so text saved on booth control applies from the next one.
    """
    from google import genai
    from google.genai import types

    # Pinned like the live socket: google-genai 3.x changed the default API version.
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))

    async def clean(line: str, shown: list[str]) -> str:
        config = types.GenerateContentConfig(
            system_instruction=instruction_with(context() if context is not None else ""),
            temperature=0.0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = await client.aio.models.generate_content(
            model=model, contents=prompt_for(line, shown), config=config
        )
        return response.text or ""

    return clean
