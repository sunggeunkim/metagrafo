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
