from __future__ import annotations

import io
import json
import logging
import wave
from collections.abc import Callable

logger = logging.getLogger(__name__)

LANGUAGE_NAMES = {"en": "English", "ko": "Korean"}

# Gemini's structured output: every reply is this object.
REPLY_SCHEMA = {
    "type": "object",
    "properties": {"heard": {"type": "string"}, "caption": {"type": "string"}},
    "required": ["heard", "caption"],
}

CAPTION_RULES = (
    "You write live captions for a church service. The audio is one stretch of speech, cut where "
    "the speaker paused, so it can start or stop in the middle of a sentence. "
    'Put exactly what was said in "heard", in the language it was spoken. Put the caption in '
    '"caption": what was said, as natural, fluent speech in the caption language, without '
    "disfluencies, false starts or filler words such as um and uh. Keep every idea, name, detail "
    "and scripture reference that was said, and add nothing that was not. Where the speech stops "
    "in the middle of a sentence, the caption stops at the same place. "
    'If there is no speech in the audio, or only noise or music, reply with "heard" and '
    '"caption" both empty. '
    'The lines under "Shown before:" are already on screen; use them only to understand the '
    "speech and never repeat them."
)

# Last in the instruction, as tested on 2026-11-15: without it gemini-4-hydrogen made a sermon
# line out of 1 s of silence (3 of 3 tries); with it, silence, noise and music came back empty
# (9 of 9) and speech was still captioned. The same wording made gemini-3.8-flash worse.
NEVER_INVENT = (
    "\n\nMost important: the audio is often silence, room noise, music or singing with no one "
    "talking. Only write words you clearly hear a person say in this audio. Never guess, never "
    "continue the sermon from the lines shown before, and never make up a sentence that sounds "
    'like a sermon. If you cannot clearly hear spoken words, "heard" and "caption" are both '
    "empty. An empty caption is always better than words nobody said."
)
PROMPT = "Write the caption for this audio. If no one is clearly speaking in it, reply with both fields empty."


def instruction_for(
    source: str,
    target: str,
    *,
    context: str = "",
    korean_style: str = "",
    glossary: str = "",
) -> str:
    """What the model is told for one piece: the languages, then the booth's style, glossary
    and notes on this service, when set. The Korean style goes only with Korean captions."""
    spoken, caption = LANGUAGE_NAMES.get(source, source), LANGUAGE_NAMES.get(target, target)
    instruction = f"{CAPTION_RULES}\n\nThe speech is spoken in {spoken}. Write the caption in {caption}."
    if source == target:
        instruction += " It is a cleaned-up transcript: never translate it."
    if target == "ko" and korean_style.strip():
        instruction += f"\n\nWrite the Korean caption in this style:\n{korean_style.strip()}"
    if glossary.strip():
        instruction += (
            "\n\nGlossary from the booth. When one of these terms is meant, use the wording given "
            f"here, in the caption's language:\n{glossary.strip()}"
        )
    if context.strip():
        instruction += (
            "\n\nNotes about this service, from the booth. Follow any instructions there about "
            "wording or style. Use the rest only to get names, scripture references and terms "
            f"right; never add it to the caption:\n{context.strip()}"
        )
    return instruction + NEVER_INVENT


def _wav(pcm_s16le: bytes, sample_rate: int = 16000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_s16le)
    return out.getvalue()


def make_gemini_pauses_transcribe(
    api_key: str,
    model: str,
    *,
    thinking_budget: int = 0,
    timeout_s: float = 15.0,
    context: Callable[[], str] | None = None,
    korean_style: Callable[[], str] | None = None,
    glossary: Callable[[], str] | None = None,
):
    """PCM in, caption out: a Gemini text model hears one piece cut at a pause and writes its
    caption in one call. Source and target come from the caller's language and task, as for
    Hermeneia. `context`, `korean_style` and `glossary` are read for every piece."""
    from google import genai
    from google.genai import types

    # Pinned like cleanup and the live socket: google-genai 3.x changed the default API version.
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta", timeout=int(timeout_s * 1000)),
    )

    def transcribe(
        pcm_s16le: bytes,
        *,
        language: str,
        task: str,
        initial_prompt: str,
        condition_on_previous_text: bool = False,
        target_language: str | None = None,
    ) -> str:
        target = target_language or ("en" if task == "translate" else language)
        instruction = instruction_for(
            language,
            target,
            context=context() if context is not None else "",
            korean_style=korean_style() if korean_style is not None else "",
            glossary=glossary() if glossary is not None else "",
        )
        prompt = PROMPT
        if initial_prompt.strip():
            prompt = f"Shown before:\n{initial_prompt.strip()}\n\n{prompt}"
        config = types.GenerateContentConfig(
            system_instruction=instruction,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            response_mime_type="application/json",
            response_json_schema=REPLY_SCHEMA,
        )
        audio = types.Part.from_bytes(data=_wav(pcm_s16le), mime_type="audio/wav")
        try:
            response = client.models.generate_content(model=model, contents=[audio, prompt], config=config)
        except Exception as error:
            # A failed piece shows nothing, like a Hermeneia error; the next piece tries again.
            logger.warning("gemini on pauses failed: %s", str(error)[:300])
            return ""
        try:
            reply = json.loads(response.text or "")
            heard, caption = str(reply["heard"]), str(reply["caption"])
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("gemini on pauses reply is not the JSON asked for: %r", (response.text or "")[:300])
            return ""
        logger.info("gemini on pauses heard: %s", heard)
        return caption.strip()

    return transcribe
