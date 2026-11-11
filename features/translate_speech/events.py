from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SubtitleEvent:
    chunk_id: str
    text: str
    source_language: str
    target_language: str
    mode: str
    created_at: float
    duration_s: float
    provisional: bool = False
    # Wall clock (time.time()) when the speech began; None when unknown.
    spoken_at: float | None = None
    # A finished line the overlay keeps up long enough to read (Gemini live, Whole sentences).
    paced: bool = False
    # The line as heard, when a text model rewrote it before it was shown.
    before_cleanup: str | None = None


@dataclass(frozen=True, slots=True)
class SpeechHeardEvent:
    """A piece of what the speaker said, as Gemini live transcribed it, before translation."""

    text: str
    language: str
    heard_at: float
