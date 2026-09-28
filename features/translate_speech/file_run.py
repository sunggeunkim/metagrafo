from __future__ import annotations

from pathlib import Path

from features.capture_audio.chunker import SpeechChunker
from features.capture_audio.wav import frames_from_wav
from features.operator_control.events import TranslateMode
from features.translate_speech.captioner import Captioner


def run_wav(
    *,
    wav_path: Path,
    out_path: Path,
    transcribe,
    chunker: SpeechChunker,
    mode: str,
    initial_prompt: str,
    min_silence_ms: int,
    engine: str = "whisper",
    translate=None,
    start_s: float = 0.0,
    end_s: float | None = None,
    condition_on_previous_text: bool = False,
) -> int:
    try:
        translate_mode = TranslateMode(mode)
    except ValueError:
        translate_mode = TranslateMode.KO_TO_EN
    captioner = Captioner(
        transcribe,
        engine=engine,
        translate=translate,
        glossary=initial_prompt,
        condition_on_previous_text=condition_on_previous_text,
    )

    lines: list[str] = []

    def _take(parts: list[bytes]) -> None:
        for pcm in parts:
            text = captioner.line(pcm, direction=translate_mode.value)
            cleaned = (text or "").strip()
            if cleaned:
                lines.append(cleaned)

    for frame in frames_from_wav(
        wav_path,
        min_silence_ms=min_silence_ms,
        start_s=start_s,
        end_s=end_s,
    ):
        _take(chunker.push(frame))
    _take(chunker.flush())

    out_path = Path(out_path)
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    out_path.write_text(payload, encoding="utf-8")
    return len(lines)
