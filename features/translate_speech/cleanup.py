from __future__ import annotations

import json
from collections.abc import Callable

from core.settings import GLOSSARY_DEFAULT, KOREAN_STYLE_DEFAULT
from features.translate_speech.gemini_live import Cleaned, Cleanup, is_korean

CLEANUP_RULES = (
    "You clean up live captions of a church sermon. They were machine-translated while the "
    "pastor was still speaking, so they can backtrack or read awkwardly. Rewrite the line under "
    '"Clean up:" so it reads as natural, fluent speech in its language, the way a native speaker '
    "would say it: remove disfluencies and false starts, fix grammar, and smooth awkward wording "
    "and word order. Smooth how it is said, never what it says: every idea, name and detail of "
    "the line stays in it, and nothing is added. Keep scripture references and names as spoken. "
    "Keep the line's language. "
    'The lines under "Shown before:" are already on screen; use them only to understand the '
    "line and never repeat them.\n\n"
    'Under "Said:" is what the speaker actually said around this line, in the language they '
    "spoke; it may include words from the lines just before or after. Use it to correct words "
    "the translation got wrong, such as a wrong term or a meaning turned around, but never add "
    "what belongs to another line.\n\n"
)

# How a line is split when its end may wait for the next line.
SPLIT_RULES = (
    "The line was cut wherever the translation paused, so it can end in the middle of a "
    'sentence. Reply in JSON. In "finished", put the cleaned-up complete sentences of the line. '
    'In "unfinished", put the words at the end of the line that do not finish a sentence yet, '
    "copied exactly as they are in the line: they start the next line and are cleaned there, "
    'once the sentence is complete. Leave them out of "finished", and never finish them with '
    'words from "Said:" or "Shown before:". When the whole line is complete, "unfinished" is '
    'empty; when none of it is, "finished" is empty.'
)

# For an English line, instead of the Korean style: with the style and the glossary's Korean in
# the instruction, the model answered English captions in Korean.
ENGLISH_LINE = (
    "\n\nThe line is English, so the reply is English: clean it up in English and never translate "
    "it. Use only the English side of the glossary."
)

INSTRUCTION = CLEANUP_RULES + SPLIT_RULES

# In place of SPLIT_RULES, for a line that starts with words held over from the line before:
# they have waited once, so all of it shows now.
SHOW_WHOLE = (
    "This line shows now, even if it ends in the middle of a sentence: its first words already "
    'waited for it. Reply in JSON. Put all of the line, cleaned up, in "finished", including the '
    "words at its end that do not finish a sentence: every part of the line is shown, so leave "
    'nothing out. "unfinished" is always empty. Where the line stops in the middle of a '
    'sentence, "finished" stops at the same place, on the line\'s own last words, with no word or '
    "ending added to finish them and nothing changed in meaning to make it sound complete. An "
    "ending the line already has still follows the Korean style, if any."
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
    context: str,
    korean_style: str = KOREAN_STYLE_DEFAULT,
    glossary: str = GLOSSARY_DEFAULT,
    *,
    hold: bool = True,
) -> str:
    """The instruction, plus the booth's Korean style, glossary and notes on this service, when set.
    Without `hold`, the line shows whole: SHOW_WHOLE replaces the split rules."""
    instruction = INSTRUCTION if hold else CLEANUP_RULES + SHOW_WHOLE
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

    async def clean(line: str, shown: list[str], said: str = "", *, hold: bool = True) -> Cleaned:
        korean = is_korean(line)
        style = korean_style() if korean_style is not None else KOREAN_STYLE_DEFAULT
        instruction = instruction_with(
            context() if context is not None else "",
            style if korean else "",
            glossary() if glossary is not None else GLOSSARY_DEFAULT,
            hold=hold,
        )
        config = types.GenerateContentConfig(
            system_instruction=instruction if korean else instruction + ENGLISH_LINE,
            temperature=0.0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            response_mime_type="application/json",
            response_json_schema=REPLY_SCHEMA,
        )
        response = await client.aio.models.generate_content(
            model=model, contents=prompt_for(line, shown, said), config=config
        )
        # A reply that isn't this object raises, and the line shows as heard.
        try:
            reply = json.loads(response.text or "")
        except json.JSONDecodeError:
            raise ValueError(f"reply is not JSON: {(response.text or '')[:300]!r}") from None
        return Cleaned(finished=str(reply["finished"]), unfinished=str(reply["unfinished"]))

    return clean
