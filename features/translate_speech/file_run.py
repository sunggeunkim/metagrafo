from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from features.capture_audio.chunker import SpeechChunker
from features.capture_audio.wav import FRAME_SAMPLES, TARGET_RATE, frames_from_wav
from features.operator_control.events import TranslateMode
from features.translate_speech.captioner import Captioner

BYTES_PER_S = TARGET_RATE * 2
# As the booth's TranslateWorker: with `join_short`, pieces under 2 s wait for the next, and
# go alone when no next piece comes within 1.5 s.
JOIN_UNDER_S = 2.0
HOLD_S = 1.5


@dataclass(frozen=True, slots=True)
class Cue:
    """One caption and the speech it came from, in seconds into the recording. `end_s` is where
    the speech stopped, without the pause that ended the piece."""

    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True, slots=True)
class _Piece:
    pcm: bytes
    start_s: float
    speech_end_s: float  # without the pause that ended it
    cut_at_s: float  # when the chunker handed it over


def run_wav(
    *,
    wav_path: Path,
    out_path: Path,
    transcribe,
    chunker: SpeechChunker,
    mode: str,
    min_silence_ms: int,
    engine: str = "whisper",
    translate=None,
    start_s: float = 0.0,
    end_s: float | None = None,
    condition_on_previous_text: bool = False,
    join_short: bool = False,
) -> list[Cue]:
    """Caption a wav piece by piece, as the booth would, and write one line per caption.
    The caller picks a mode the engine can run: en_to_ko only on an engine that writes Korean."""
    try:
        translate_mode = TranslateMode(mode)
    except ValueError:
        translate_mode = TranslateMode.KO_TO_EN
    captioner = Captioner(
        transcribe,
        engine=engine,
        translate=translate,
        condition_on_previous_text=condition_on_previous_text,
    )
    frame_s = FRAME_SAMPLES / TARGET_RATE
    cues: list[Cue] = []
    held: list[_Piece] = []

    def _caption(pieces: list[_Piece]) -> None:
        text = captioner.line(b"".join(p.pcm for p in pieces), direction=translate_mode.value)
        cleaned = (text or "").strip()
        if cleaned:
            cues.append(Cue(pieces[0].start_s, pieces[-1].speech_end_s, cleaned))

    def _take(parts: list[bytes], cut_at_s: float) -> None:
        if not parts:
            return
        # Timed from the frames each piece holds: with two pauses, the silence bridged between
        # the phrases is in neither, and a 12 s cut has no pause at its end.
        spans = chunker.emitted_spans
        assert len(spans) == len(parts)
        pieces = [
            _Piece(
                pcm,
                start_s + span.start_frame * frame_s,
                start_s + (span.start_frame + span.frames - span.trailing_frames) * frame_s,
                cut_at_s,
            )
            for pcm, span in zip(parts, spans, strict=True)
        ]
        for piece in pieces:
            if not join_short:
                _caption([piece])
                continue
            if held and piece.cut_at_s - held[0].cut_at_s > HOLD_S:
                _caption(held)
                held.clear()
            held.append(piece)
            if sum(len(p.pcm) for p in held) / BYTES_PER_S >= JOIN_UNDER_S:
                _caption(held)
                held.clear()

    pushed = 0
    for frame in frames_from_wav(
        wav_path,
        min_silence_ms=min_silence_ms,
        start_s=start_s,
        end_s=end_s,
    ):
        pushed += 1
        _take(chunker.push(frame), start_s + pushed * frame_s)
    _take(chunker.flush(), start_s + pushed * frame_s)
    if held:
        _caption(held)

    out_path = Path(out_path)
    payload = "\n".join(cue.text for cue in cues)
    if payload:
        payload += "\n"
    out_path.write_text(payload, encoding="utf-8")
    return cues
