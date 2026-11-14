from dataclasses import dataclass
from enum import StrEnum


class TranslateMode(StrEnum):
    KO_TO_EN = "ko_to_en"
    EN_TO_EN = "en_to_en"
    EN_TO_KO = "en_to_ko"


@dataclass(frozen=True, slots=True)
class CaptionsStateEvent:
    is_active: bool


@dataclass(frozen=True, slots=True)
class ModeChangedEvent:
    mode: str
    source_language: str
    target_language: str


class OverlayPosition(StrEnum):
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"


@dataclass(frozen=True, slots=True)
class VadSilenceMsEvent:
    vad_min_silence_ms: int


@dataclass(frozen=True, slots=True)
class VadPausesEvent:
    vad_pauses: int


@dataclass(frozen=True, slots=True)
class ConditionOnPreviousTextEvent:
    condition_on_previous_text: bool


@dataclass(frozen=True, slots=True)
class WholeSentencesEvent:
    """Gemini live shows only finished lines. Takes effect at the next line."""

    whole_sentences: bool


@dataclass(frozen=True, slots=True)
class OneSentenceEvent:
    """The overlay shows only the newest sentence, whole; off keeps rolling rows."""

    one_sentence: bool


@dataclass(frozen=True, slots=True)
class CleanUpWordingEvent:
    """A text model rewrites each finished Gemini live line. Takes effect at the next line."""

    clean_up_wording: bool


@dataclass(frozen=True, slots=True)
class CleanupContextEvent:
    """Background on this service for the cleanup model. Takes effect at the next cleaned line."""

    text: str


@dataclass(frozen=True, slots=True)
class KoreanStyleEvent:
    """How cleanup writes Korean lines; empty means no style rule. From the next cleaned line."""

    text: str


@dataclass(frozen=True, slots=True)
class GlossaryEvent:
    """Church terms cleanup keeps to; empty means no glossary. From the next cleaned line."""

    text: str


@dataclass(frozen=True, slots=True)
class SaveTranscriptEvent:
    """Write each caption, with its timing, to a file for later analysis. On starts a new file."""

    save_transcript: bool


@dataclass(frozen=True, slots=True)
class GeminiKeyChangedEvent:
    """The operator saved a Gemini key. The key itself stays on Settings."""


@dataclass(frozen=True, slots=True)
class CaptionEngineChangedEvent:
    """The operator picked an engine or saved a Hermeneia token. Both stay on Settings."""


@dataclass(frozen=True, slots=True)
class YouTubeCaptionUrlChangedEvent:
    """The operator saved or cleared the YouTube caption URL. The URL stays on Settings."""


@dataclass(frozen=True, slots=True)
class AudioDeviceChangedEvent:
    """The operator picked a recording device. The name stays on Settings."""


@dataclass(frozen=True, slots=True)
class OverlayStyleEvent:
    position: str
    font_size_vw: float
    inset_vertical_pct: float
    inset_horizontal_pct: float
    box_width_pct: float
    box_height_pct: float
    line_count: int


def languages_for(mode: TranslateMode) -> tuple[str, str]:
    if mode is TranslateMode.EN_TO_EN:
        return "en", "en"
    if mode is TranslateMode.EN_TO_KO:
        return "en", "ko"
    return "ko", "en"


def korean_captions(mode: TranslateMode) -> bool:
    """The mode puts Korean on screen. Whisper only translates into English, so only
    Gemini live and Seamless on Hermeneia write it (`Settings.writes_korean`)."""
    return mode is TranslateMode.EN_TO_KO
