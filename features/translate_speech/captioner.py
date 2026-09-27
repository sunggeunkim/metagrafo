from __future__ import annotations

from collections.abc import Callable

Recognize = Callable[..., str]
Translate = Callable[..., str]


class Captioner:
    def __init__(
        self,
        recognize: Recognize,
        *,
        engine: str = "whisper",
        translate: Translate | None = None,
        glossary: str = "",
    ) -> None:
        self._recognize = recognize
        self._engine = engine
        self._translate = translate
        self._glossary = glossary
        self._prior_english: list[str] = []

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
            initial_prompt=self._glossary,
        )
        text = (recognized or "").strip()
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
