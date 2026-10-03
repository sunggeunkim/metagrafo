from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioChunkEvent:
    chunk_id: str
    pcm_s16le: bytes
    sample_rate: int
    started_at: float
    duration_s: float


@dataclass(frozen=True, slots=True)
class ProgramAudioEvent:
    """About 100 ms of 16 kHz mono program audio, before a VAD cut."""

    pcm_s16le: bytes
    sample_rate: int
