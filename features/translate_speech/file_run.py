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
    )

    lines: list[str] = []
    for frame in frames_from_wav(wav_path, min_silence_ms=min_silence_ms):
        pcm = chunker.push(frame)
        if not pcm:
            continue
        text = captioner.line(pcm, direction=translate_mode.value)
        cleaned = (text or "").strip()
        if cleaned:
            lines.append(cleaned)

    out_path = Path(out_path)
    payload = "\n".join(lines)
    if payload:
        payload += "\n"
    out_path.write_text(payload, encoding="utf-8")
    return len(lines)
