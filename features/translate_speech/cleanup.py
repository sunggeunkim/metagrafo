from __future__ import annotations

from features.translate_speech.gemini_live import Cleanup

# The newest flash-lite text model that answers inside the 1.5 s cleanup budget (3.5 Flash Lite
# took 1–8 s per line in a booth test on 2026-11-09; 3.1 Flash Lite about 0.5 s).
CLEANUP_MODEL = "gemini-3.1-flash-lite"

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


def gemini_cleanup(api_key: str, model: str = CLEANUP_MODEL) -> Cleanup:
    """A fast Gemini text model that rewrites one finished caption line."""
    from google import genai
    from google.genai import types

    # Pinned like the live socket: google-genai 3.x changed the default API version.
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
    config = types.GenerateContentConfig(
        system_instruction=INSTRUCTION,
        temperature=0.0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    async def clean(line: str, shown: list[str]) -> str:
        response = await client.aio.models.generate_content(
            model=model, contents=prompt_for(line, shown), config=config
        )
        return response.text or ""

    return clean
