from __future__ import annotations

from collections.abc import Callable

Recognize = Callable[..., str]
Translate = Callable[..., str]


_PROMPT_TAIL = 800


class Captioner:
    def __init__(
        self,
        recognize: Recognize,
        *,
        engine: str = "whisper",
        translate: Translate | None = None,
        glossary: str = "",
        condition_on_previous_text: bool = False,
    ) -> None:
        self._recognize = recognize
        self._engine = engine
        self._translate = translate
        self._glossary = glossary
        self._prior_english: list[str] = []
        self._condition_on_previous_text = condition_on_previous_text
        self._previous = ""

    def set_condition_on_previous_text(self, enabled: bool) -> None:
        self._condition_on_previous_text = enabled

    def clear_previous(self) -> None:
        self._previous = ""

    def _prompt(self) -> str:
        glossary = self._glossary.strip()
        if not self._condition_on_previous_text or not self._previous:
            return glossary
        room = _PROMPT_TAIL - len(glossary) - 1
        if room < 1:
            return glossary
        tail = self._previous if len(self._previous) <= room else self._previous[-room:]
        if not glossary:
            return tail.strip()
        return f"{tail.strip()} {glossary}"

    def _remember(self, text: str) -> None:
        self._previous = f"{self._previous} {text}".strip()
        if len(self._previous) > 2000:
            self._previous = self._previous[-2000:]

    def line(self, pcm: bytes, *, direction: str) -> str:
        if direction == "en_to_en":
            language, task = "en", "transcribe"
        elif self._engine == "gemini":
            language, task = "ko", "transcribe"
        else:
            language, task = "ko", "translate"
        recognized = self._recognize(
            pcm,
            language=language,
            task=task,
            initial_prompt=self._prompt(),
            condition_on_previous_text=self._condition_on_previous_text,
        )
        text = (recognized or "").strip()
        if self._condition_on_previous_text and text:
            self._remember(text)
        if direction == "en_to_en" or self._engine != "gemini":
            return text
        if not text or self._translate is None:
            return ""
        translated = self._translate(
            text,
            prior_english=list(self._prior_english),
            glossary=self._glossary,
        )
        translated = (translated or "").strip()
        if translated:
            self._prior_english.append(translated)
            self._prior_english = self._prior_english[-4:]
        return translated
