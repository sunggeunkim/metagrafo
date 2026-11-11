from __future__ import annotations

import json
from collections.abc import Callable

from core.settings import GLOSSARY_DEFAULT, KOREAN_STYLE_DEFAULT
from features.translate_speech.gemini_live import Cleaned, Cleanup

INSTRUCTION = (
    "You clean up live captions of a church sermon. They were machine-translated while the "
    "pastor was still speaking, so they can backtrack or read awkwardly. Rewrite the line under "
    '"Clean up:": remove disfluencies and false starts, lightly fix grammar, and keep the meaning. '
    "Add nothing. Keep scripture references and names as spoken. Keep the line's language. "
    'The lines under "Shown before:" are already on screen; use them only to understand the '
    "line and never repeat them.\n\n"
    'Under "Said:" is what the speaker actually said around this line, in the language they '
    "spoke; it may include words from the lines just before or after. Use it to correct words "
    "the translation got wrong, such as a wrong term or a meaning turned around, but never add "
    "what belongs to another line.\n\n"
    "The line was cut wherever the translation paused, so it can end in the middle of a "
    'sentence. Reply in JSON. In "finished", put the cleaned-up complete sentences of the line. '
    'In "unfinished", put the words at the end of the line that do not finish a sentence yet, '
    "copied exactly as they are in the line: they start the next line and are cleaned there, "
    'once the sentence is complete. Leave them out of "finished", and never finish them with '
    'words from "Said:" or "Shown before:". When the whole line is complete, "unfinished" is '
    'empty; when none of it is, "finished" is empty.'
)

# Gemini's structured output: every reply is this object.
REPLY_SCHEMA = {
    "type": "object",
    "properties": {"finished": {"type": "string"}, "unfinished": {"type": "string"}},
    "required": ["finished", "unfinished"],
}


def prompt_for(line: str, shown: list[str], said: str = "") -> str:
    parts = []
    if shown:
        parts.append("Shown before:\n" + "".join(f"{text}\n" for text in shown))
    if said.strip():
        parts.append(f"Said:\n{said.strip()}\n")
    parts.append(f"Clean up:\n{line}")
    return "\n".join(parts)


def instruction_with(
    context: str, korean_style: str = KOREAN_STYLE_DEFAULT, glossary: str = GLOSSARY_DEFAULT
) -> str:
    """The instruction, plus the booth's Korean style, glossary and notes on this service, when set."""
    instruction = INSTRUCTION
    if korean_style.strip():
        instruction += (
            "\n\nWhen the line is Korean, follow this style. Changing the endings this way is part "
            f"of cleaning up; still keep the meaning and add nothing:\n{korean_style.strip()}"
        )
    if glossary.strip():
        instruction += (
            "\n\nGlossary from the booth. When one of these terms is meant, use the wording given "
            f"here, whichever way the line was translated:\n{glossary.strip()}"
        )
    if context.strip():
        instruction += (
            "\n\nNotes about this service, from the booth. Follow any instructions there about "
            "wording or style. Use the rest only to get names, scripture references and terms "
            f"right; never add it to the line:\n{context.strip()}"
        )
    return instruction


def gemini_cleanup(
    api_key: str,
    model: str,
    context: Callable[[], str] | None = None,
    korean_style: Callable[[], str] | None = None,
    glossary: Callable[[], str] | None = None,
) -> Cleanup:
    """A fast Gemini text model (`CLEANUP_MODEL`) that rewrites one finished caption line.

    `context`, `korean_style` and `glossary` are read for every line, so text saved on booth
    control applies from the next one. Without `korean_style` or `glossary`, the defaults apply.
    """
    from google import genai
    from google.genai import types

    # Pinned like the live socket: google-genai 3.x changed the default API version.
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))

    async def clean(line: str, shown: list[str], said: str = "") -> Cleaned:
        config = types.GenerateContentConfig(
            system_instruction=instruction_with(
                context() if context is not None else "",
                korean_style() if korean_style is not None else KOREAN_STYLE_DEFAULT,
                glossary() if glossary is not None else GLOSSARY_DEFAULT,
            ),
            temperature=0.0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            response_mime_type="application/json",
            response_json_schema=REPLY_SCHEMA,
        )
        response = await client.aio.models.generate_content(
            model=model, contents=prompt_for(line, shown, said), config=config
        )
        # A reply that isn't this object raises, and the line shows as heard.
        reply = json.loads(response.text or "")
        return Cleaned(finished=str(reply["finished"]), unfinished=str(reply["unfinished"]))

    return clean
