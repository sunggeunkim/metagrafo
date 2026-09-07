from __future__ import annotations

import logging

import numpy as np

from features.translate_speech.device_profile import WhisperProfile

logger = logging.getLogger(__name__)


def make_transcribe(profile: WhisperProfile):
    from faster_whisper import WhisperModel

    model = None
    attempts = [(profile.device, profile.compute_type)]
    if profile.device == "cuda" and profile.compute_type == "int8":
        attempts.append(("cuda", "int8_float16"))
        attempts.append(("cpu", "int8"))
    last_error: Exception | None = None
    for device, compute in attempts:
        try:
            model = WhisperModel(profile.model, device=device, compute_type=compute)
            logger.info("loaded whisper %s device=%s compute=%s", profile.model, device, compute)
            break
        except Exception as exc:
            last_error = exc
            logger.warning("whisper load failed device=%s compute=%s: %s", device, compute, exc)
    if model is None:
        raise RuntimeError("could not load faster-whisper") from last_error

    def transcribe(
        pcm_s16le: bytes,
        *,
        language: str,
        task: str,
        initial_prompt: str,
    ) -> str:
        audio = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _info = model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=initial_prompt or None,
        )
        return "".join(segment.text for segment in segments).strip()

    return transcribe
