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
        condition_on_previous_text: bool = False,
    ) -> None:
        self._recognize = recognize
        self._engine = engine
        self._translate = translate
        self._prior_english: list[str] = []
        self._condition_on_previous_text = condition_on_previous_text
        self._previous = ""

    def use_recognize(self, recognize: Recognize) -> None:
        self._recognize = recognize

    def set_condition_on_previous_text(self, enabled: bool) -> None:
        self._condition_on_previous_text = enabled

    def clear_previous(self) -> None:
        self._previous = ""

    def _prompt(self) -> str:
        if not self._condition_on_previous_text or not self._previous:
            return ""
        if len(self._previous) <= _PROMPT_TAIL:
            return self._previous
        return self._previous[-_PROMPT_TAIL:]

    def _remember(self, text: str) -> None:
        self._previous = f"{self._previous} {text}".strip()
        if len(self._previous) > 2000:
            self._previous = self._previous[-2000:]

    def line(self, pcm: bytes, *, direction: str) -> str:
        korean: dict[str, str] = {}
        if direction == "en_to_en":
            language, task = "en", "transcribe"
        elif direction == "en_to_ko":
            # Only Seamless on Hermeneia is handed en_to_ko: it takes the target as asked.
            language, task, korean = "en", "translate", {"target_language": "ko"}
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
            **korean,
        )
        text = (recognized or "").strip()
        if self._condition_on_previous_text and text:
            self._remember(text)
        if direction in ("en_to_en", "en_to_ko") or self._engine != "gemini":
            return text
        if not text or self._translate is None:
            return ""
        translated = self._translate(
            text,
            prior_english=list(self._prior_english),
        )
        translated = (translated or "").strip()
        if translated:
            self._prior_english.append(translated)
            self._prior_english = self._prior_english[-4:]
        return translated
