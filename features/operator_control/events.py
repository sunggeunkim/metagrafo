from dataclasses import dataclass
from enum import StrEnum


class TranslateMode(StrEnum):
    KO_TO_EN = "ko_to_en"
    EN_TO_EN = "en_to_en"


@dataclass(frozen=True, slots=True)
class CaptionsStateEvent:
    is_active: bool


@dataclass(frozen=True, slots=True)
class ModeChangedEvent:
    mode: str
    source_language: str
    target_language: str


def languages_for(mode: TranslateMode) -> tuple[str, str]:
    if mode is TranslateMode.EN_TO_EN:
        return "en", "en"
    return "ko", "en"
