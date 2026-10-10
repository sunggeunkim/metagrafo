from features.translate_speech.captioner import Captioner

PCM = b"\x00\x10" * 160


def test_whisper_korean_caption_is_the_english_translation() -> None:
    def recognize(
        _pcm: bytes,
        *,
        language: str,
        task: str,
        initial_prompt: str,
        condition_on_previous_text: bool = False,
    ) -> str:
        if language == "ko" and task == "translate" and initial_prompt == "":
            return "God is good."
        return "하나님은 선하시다."

    captioner = Captioner(recognize)
    assert captioner.line(PCM, direction="ko_to_en") == "God is good."


def test_whisper_english_guest_caption_is_the_transcript() -> None:
    def recognize(
        _pcm: bytes,
        *,
        language: str,
        task: str,
        initial_prompt: str,
        condition_on_previous_text: bool = False,
    ) -> str:
        if language == "en" and task == "transcribe" and initial_prompt == "":
            return "Hello, church."
        return "translated by mistake"

    captioner = Captioner(recognize)
    assert captioner.line(PCM, direction="en_to_en") == "Hello, church."


def test_gemini_korean_caption_uses_prior_english_not_korean() -> None:
    def recognize(
        _pcm: bytes,
        *,
        language: str,
        task: str,
        initial_prompt: str,
        condition_on_previous_text: bool = False,
    ) -> str:
        if language == "ko" and task == "transcribe" and initial_prompt == "":
            return "한나" if _pcm == b"next" else "기도"
        return "God is good."

    def translate(source: str, *, prior_english: list[str]) -> str:
        if source == "기도" and prior_english == []:
            return "Prayer"
        if source == "한나" and prior_english == ["Prayer"]:
            return "Hannah"
        return "bad context"

    captioner = Captioner(recognize, engine="gemini", translate=translate)
    assert captioner.line(PCM, direction="ko_to_en") == "Prayer"
    assert captioner.line(b"next", direction="ko_to_en") == "Hannah"


def test_gemini_english_guest_is_not_translated() -> None:
    def recognize(
        _pcm: bytes,
        *,
        language: str,
        task: str,
        initial_prompt: str,
        condition_on_previous_text: bool = False,
    ) -> str:
        if language == "en" and task == "transcribe":
            return "Hello, church."
        return "기도"

    def translate(_source: str, *, prior_english: list[str]) -> str:
        raise AssertionError("gemini should not run for an english guest")

    captioner = Captioner(recognize, engine="gemini", translate=translate)
    assert captioner.line(PCM, direction="en_to_en") == "Hello, church."


def test_gemini_context_keeps_only_the_last_four_english_lines() -> None:
    sources = iter(["하나", "둘", "셋", "넷", "다섯", "여섯"])

    def recognize(
        _pcm: bytes,
        *,
        language: str,
        task: str,
        initial_prompt: str,
        condition_on_previous_text: bool = False,
    ) -> str:
        return next(sources)

    seen: list[list[str]] = []

    def translate(source: str, *, prior_english: list[str]) -> str:
        seen.append(list(prior_english))
        return source

    captioner = Captioner(recognize, engine="gemini", translate=translate)
    for _ in range(6):
        captioner.line(PCM, direction="ko_to_en")
    assert seen[5] == ["둘", "셋", "넷", "다섯"]


def test_previous_text_stays_off_until_asked() -> None:
    seen: list[tuple[str, bool]] = []

    def recognize(
        _pcm: bytes,
        *,
        language: str,
        task: str,
        initial_prompt: str,
        condition_on_previous_text: bool = False,
    ) -> str:
        seen.append((initial_prompt, condition_on_previous_text))
        return "Hannah"

    captioner = Captioner(recognize)
    captioner.line(PCM, direction="ko_to_en")
    captioner.line(PCM, direction="ko_to_en")
    assert seen == [("", False), ("", False)]


def test_previous_text_is_given_to_the_next_phrase() -> None:
    seen: list[tuple[str, bool]] = []
    lines = iter(["Hannah", "prayed"])

    def recognize(
        _pcm: bytes,
        *,
        language: str,
        task: str,
        initial_prompt: str,
        condition_on_previous_text: bool = False,
    ) -> str:
        seen.append((initial_prompt, condition_on_previous_text))
        return next(lines)

    captioner = Captioner(recognize, condition_on_previous_text=True)
    assert captioner.line(PCM, direction="ko_to_en") == "Hannah"
    assert captioner.line(PCM, direction="ko_to_en") == "prayed"
    assert seen[0] == ("", True)
    assert seen[1] == ("Hannah", True)
