from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    """Where one handed-over piece sits among the frames pushed (counted from 0), and how many
    of its frames at the end are the pause that ended it (0 when the 12 s cap cut it)."""

    start_frame: int
    frames: int
    trailing_frames: int


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
        pauses_to_cut: int = 1,
        is_speech: Callable[[bytes], bool],
    ) -> None:
        self._is_speech = is_speech
        self._frame_ms = frame_samples / sample_rate * 1000
        frame_ms = self._frame_ms
        self._preroll_frames = max(1, int(round(preroll_ms / frame_ms))) if preroll_ms else 0
        self.set_min_silence_ms(min_silence_ms)
        self.set_pauses_to_cut(pauses_to_cut)
        self._min_speech_frames = max(1, int(round(min_speech_ms / frame_ms)))
        self._max_frames = max(1, int(round(max_utterance_s * sample_rate / frame_samples)))
        self._preroll: deque[bytes] = deque(maxlen=self._preroll_frames or 1)
        self._utterance: list[bytes] = []
        self._phrases: list[list[bytes]] = []
        self._in_speech = False
        self._trailing_silence = 0
        self._pauses_seen = 0
        # After a pause that is not yet enough to cut, further silence is not
        # kept in the phrase. The 12 s cap still counts that wait and emits.
        self._bridging = False
        self._wait_frames = 0
        # Where each piece came from, for a file run that times its captions. Bridged silence
        # between two phrases is in neither, so the pieces are not back to back.
        self._frame_index = -1
        self._utterance_start = 0
        self._phrase_spans: list[tuple[int, int]] = []
        self.emitted_spans: list[ChunkSpan] = []

    def push(self, frame: bytes) -> list[bytes]:
        self._frame_index += 1
        if self._bridging and self._pauses_seen >= self._pauses_to_cut:
            return self._emit()

        voiced = self._is_speech(frame)
        if self._bridging and not voiced:
            self._wait_frames += 1
            if self._preroll_frames:
                self._preroll.append(frame)
            if self._held_frames() >= self._max_frames:
                return self._emit()
            return []
        if self._bridging:
            self._bridging = False
            self._wait_frames = 0
            self._trailing_silence = 0
            self._in_speech = False

        if not self._in_speech:
            if self._preroll_frames:
                self._preroll.append(frame)
            if not voiced:
                return []
            self._in_speech = True
            self._utterance = list(self._preroll) if self._preroll_frames else [frame]
            if self._preroll_frames and self._utterance[-1] is not frame:
                self._utterance.append(frame)
            self._utterance_start = self._frame_index - len(self._utterance) + 1
            self._trailing_silence = 0
            return []

        self._utterance.append(frame)
        if voiced:
            self._trailing_silence = 0
            if self._held_frames() >= self._max_frames:
                return self._emit(include_open=True)
            return []

        self._trailing_silence += 1
        if self._trailing_silence >= self._silence_frames:
            speech_frames = len(self._utterance) - self._trailing_silence
            if speech_frames < self._min_speech_frames:
                if self._pauses_seen == 0:
                    self.reset()
                else:
                    self._utterance = []
                    self._trailing_silence = 0
                    self._in_speech = False
                    self._bridging = True
                return []
            self._pauses_seen += 1
            self._phrases.append(list(self._utterance))
            self._phrase_spans.append((self._utterance_start, self._trailing_silence))
            self._utterance = []
            self._trailing_silence = 0
            self._in_speech = False
            if self._pauses_seen >= self._pauses_to_cut:
                return self._emit()
            self._bridging = True
            self._wait_frames = 0
            self._preroll.clear()
            return []
        if self._held_frames() >= self._max_frames:
            return self._emit(include_open=True)
        return []

    def flush(self) -> list[bytes]:
        if not self._phrases and not self._utterance:
            return []
        return self._emit(include_open=True)

    def _held_frames(self) -> int:
        held = sum(len(phrase) for phrase in self._phrases) + len(self._utterance)
        return held + self._wait_frames

    def _emit(self, *, include_open: bool = False) -> list[bytes]:
        phrases = list(self._phrases)
        spans = [
            ChunkSpan(start, len(phrase), trailing)
            for (start, trailing), phrase in zip(self._phrase_spans, self._phrases, strict=True)
        ]
        if include_open and self._utterance:
            speech_frames = len(self._utterance) - self._trailing_silence
            if speech_frames >= self._min_speech_frames:
                phrases.append(list(self._utterance))
                spans.append(ChunkSpan(self._utterance_start, len(self._utterance), self._trailing_silence))
        self.reset()
        self.emitted_spans = [span for span, phrase in zip(spans, phrases, strict=True) if phrase]
        return [b"".join(phrase) for phrase in phrases if phrase]

    def set_min_silence_ms(self, min_silence_ms: int) -> None:
        self._silence_frames = max(1, int(round(min_silence_ms / self._frame_ms)))

    def set_pauses_to_cut(self, pauses: int) -> None:
        if pauses < 1:
            pauses = 1
        elif pauses > 2:
            pauses = 2
        self._pauses_to_cut = pauses

    def reset(self) -> None:
        self._utterance = []
        self._phrases = []
        self._phrase_spans = []
        self._in_speech = False
        self._trailing_silence = 0
        self._preroll.clear()
        self._pauses_seen = 0
        self._bridging = False
        self._wait_frames = 0


