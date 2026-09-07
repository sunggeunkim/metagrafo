from __future__ import annotations

from collections import deque
from collections.abc import Callable


class SpeechChunker:
    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_samples: int = 512,
        preroll_ms: int = 200,
        min_silence_ms: int = 800,
        max_utterance_s: float = 12,
        min_speech_ms: int = 250,
        is_speech: Callable[[bytes], bool],
    ) -> None:
        self._is_speech = is_speech
        frame_ms = frame_samples / sample_rate * 1000
        self._preroll_frames = max(1, int(round(preroll_ms / frame_ms))) if preroll_ms else 0
        self._silence_frames = max(1, int(round(min_silence_ms / frame_ms)))
        self._min_speech_frames = max(1, int(round(min_speech_ms / frame_ms)))
        self._max_frames = max(1, int(round(max_utterance_s * sample_rate / frame_samples)))
        self._preroll: deque[bytes] = deque(maxlen=self._preroll_frames or 1)
        self._utterance: list[bytes] = []
        self._in_speech = False
        self._trailing_silence = 0

    def push(self, frame: bytes) -> bytes | None:
        voiced = self._is_speech(frame)
        if not self._in_speech:
            if self._preroll_frames:
                self._preroll.append(frame)
            if not voiced:
                return None
            self._in_speech = True
            self._utterance = list(self._preroll) if self._preroll_frames else [frame]
            if self._preroll_frames and self._utterance[-1] is not frame:
                self._utterance.append(frame)
            self._trailing_silence = 0
            return None

        self._utterance.append(frame)
        if voiced:
            self._trailing_silence = 0
            if len(self._utterance) >= self._max_frames:
                return self._emit()
            return None

        self._trailing_silence += 1
        if self._trailing_silence >= self._silence_frames:
            speech_frames = len(self._utterance) - self._trailing_silence
            if speech_frames < self._min_speech_frames:
                self.reset()
                return None
            return self._emit()
        if len(self._utterance) >= self._max_frames:
            return self._emit()
        return None

    def reset(self) -> None:
        self._utterance = []
        self._in_speech = False
        self._trailing_silence = 0
        self._preroll.clear()

    def _emit(self) -> bytes:
        payload = b"".join(self._utterance)
        self.reset()
        return payload
