from __future__ import annotations

import numpy as np


def energy_is_speech(frame: bytes, *, threshold: float = 500.0) -> bool:
    samples = np.frombuffer(frame, dtype=np.int16)
    if samples.size == 0:
        return False
    return float(np.abs(samples).mean()) > threshold


def load_is_speech():
    try:
        from silero_vad import load_silero_vad

        model = load_silero_vad(onnx=True)

        def is_speech(frame: bytes) -> bool:
            audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            if audio.size < 512:
                return False
            window = audio[:512]
            prob = model(window, 16000)
            value = float(prob.item() if hasattr(prob, "item") else prob)
            return value > 0.5

        return is_speech
    except Exception:
        return energy_is_speech
