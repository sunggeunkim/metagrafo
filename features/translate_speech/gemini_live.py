from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from types import TracebackType

from core.event_bus import EventBus
from features.capture_audio.events import ProgramAudioEvent
from features.operator_control.events import (
    CaptionsStateEvent,
    CleanUpWordingEvent,
    ModeChangedEvent,
    TranslateMode,
    WholeSentencesEvent,
    languages_for,
)
from features.translate_speech.events import SpeechHeardEvent, SubtitleEvent
from features.translate_speech.live_line import LiveLine
from features.translate_speech.tidy import tidy

logger = logging.getLogger(__name__)

MODEL = "gemini-3.5-live-translate-preview"
Connect = Callable[[TranslateMode], "LiveConnection"]
@dataclass(frozen=True, slots=True)
class Cleaned:
    """A text model's reply: the complete sentences to show, and the words after them that
    don't finish a sentence yet, copied unchanged from the end of the line."""

    finished: str
    unfinished: str = ""


# A finished line, the lines shown before it, and what the speaker said around it (in the
# source language) in; the line a text model rewrote out. A plain string is all finished.
# (line, shown lines, what the speaker said) -> its wording. Called with `hold=False` for a line
# that starts with words held over already: it shows whole, so nothing of it may wait again.
Cleanup = Callable[..., Awaitable["str | Cleaned"]]
SHORT_SENTENCE_CHARS = 25
# Two 1920-wide overlay rows at the default 2.5 vw hold about 140 characters. The cut leaves
# slack because the overlay can wrap a full two rows onto a third (issue #36).
LONG_LINE_CHARS = 120
CLEANUP_CONTEXT_LINES = 3
# Captions never wait longer than this for the text model; the line shows as heard instead.
CLEANUP_TIMEOUT_S = 4.0
# Words cleanup held back for the rest of their sentence wait this long for more words, not the
# usual quiet: the translation often pauses longer than that mid-sentence (2026-11-11 replay).
HELD_QUIET_S = 5.0
# The speaker's words reach the socket a little before their translation, so a line's source
# is what was heard from this long before its first translated fragment until it is committed.
SOURCE_LEAD_S = 6.0


def echoes_target_language(mode: TranslateMode) -> bool:
    """Guest mode is already English, so the socket must echo it instead of staying silent."""
    return mode is TranslateMode.EN_TO_EN


class LiveConnection:
    """The slice of a Gemini live socket this translator uses."""

    async def __aenter__(self) -> LiveConnection:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False

    def send_realtime_input(self, **kwargs: object) -> object:
        raise NotImplementedError

    def receive(self) -> AsyncIterator[object]:
        raise NotImplementedError


def gemini_connect(api_key: str) -> Connect:
    def connect(mode: TranslateMode) -> LiveConnection:
        from google import genai
        from google.genai import types

        # google-genai 3.0 defaults to an API version that 404s the Live preview model.
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            translation_config=types.TranslationConfig(
                target_language_code=languages_for(mode)[1],
                echo_target_language=echoes_target_language(mode),
            ),
        )
        return client.aio.live.connect(model=MODEL, config=config)

    return connect


class GeminiLiveTranslator:
    def __init__(
        self,
        bus: EventBus,
        *,
        connect: Connect,
        api_key: str = "",
        mode: str = "ko_to_en",
        clock: Callable[[], float] = time.time,
        whole_sentences: bool = False,
        quiet_s: float = 1.2,
        held_quiet_s: float = HELD_QUIET_S,
        clean_up_wording: bool = False,
        cleanup: Cleanup | None = None,
        cleanup_timeout_s: float = CLEANUP_TIMEOUT_S,
    ) -> None:
        self._bus = bus
        # Like whole sentences, the open line keeps the cleanup setting it started with.
        self._clean_up_wording = clean_up_wording
        self._next_clean_up_wording = clean_up_wording
        self._cleanup = cleanup
        self._cleanup_timeout_s = cleanup_timeout_s
        # The lines most recently shown, for the text model to read the new one against.
        self._shown: deque[str] = deque(maxlen=CLEANUP_CONTEXT_LINES)
        # Recent input_transcription fragments and when each was heard, for the text model.
        self._heard: deque[tuple[float, str]] = deque(maxlen=200)
        # Committed lines not shown yet, oldest first, and the task that shows them in that order.
        # They are cleaned one at a time, so words a line holds back can start the next one.
        self._waiting: deque[_Waiting] = deque()
        self._in_order: asyncio.Task[None] | None = None
        # The open line keeps the setting it started with; the booth's choice waits for the next.
        self._whole_sentences = whole_sentences
        self._next_whole_sentences = whole_sentences
        self._quiet_s = quiet_s
        self._held_quiet_s = held_quiet_s
        self._quiet: asyncio.Task[None] | None = None
        self._clock = clock
        self._connect = connect
        self._api_key = api_key
        self._mode = TranslateMode(mode)
        self._line = LiveLine()
        self._audio: asyncio.Queue[bytes] = asyncio.Queue(maxsize=10)
        self._active = True
        self._stop = asyncio.Event()
        self._resumed = asyncio.Event()
        self._reconnect = asyncio.Event()
        bus.subscribe(CaptionsStateEvent, self._on_captions)
        bus.subscribe(ModeChangedEvent, self._on_mode)
        bus.subscribe(ProgramAudioEvent, self._on_audio)
        bus.subscribe(WholeSentencesEvent, self._on_whole_sentences)
        bus.subscribe(CleanUpWordingEvent, self._on_clean_up_wording)

    async def _on_whole_sentences(self, event: WholeSentencesEvent) -> None:
        self._next_whole_sentences = event.whole_sentences

    async def _on_clean_up_wording(self, event: CleanUpWordingEvent) -> None:
        self._next_clean_up_wording = event.clean_up_wording

    def use_key(self, api_key: str, connect: Connect, *, cleanup: Cleanup | None = None) -> None:
        self._api_key = api_key
        self._connect = connect
        self._cleanup = cleanup
        self._reconnect.set()

    async def _on_mode(self, event: ModeChangedEvent) -> None:
        try:
            mode = TranslateMode(event.mode)
        except ValueError:
            mode = TranslateMode.KO_TO_EN
        if mode == self._mode:
            return
        await self._close_open_line()
        self._mode = mode
        self._reconnect.set()

    async def _on_captions(self, event: CaptionsStateEvent) -> None:
        self._active = event.is_active
        if event.is_active:
            self._stop.clear()
            self._resumed.set()
            return
        self._resumed.clear()
        self._stop.set()
        self._drain_audio()
        self._stop_quiet_wait()
        self._line.commit()
        self._drop_waiting_lines()
        self._heard.clear()

    async def _on_audio(self, event: ProgramAudioEvent) -> None:
        if not self._active:
            return
        if self._audio.full():
            try:
                self._audio.get_nowait()
            except asyncio.QueueEmpty:
                return
            logger.warning("gemini live audio queue full; dropped the oldest chunk")
        self._audio.put_nowait(event.pcm_s16le)

    def _drain_audio(self) -> None:
        while not self._audio.empty():
            try:
                self._audio.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _close_open_line(self) -> None:
        self._stop_quiet_wait()
        spoken_at = self._line.opened_at
        finished = self._finish(self._line.text) or ""
        times, carried = self._commit_line(finished)
        if finished and self._active:
            await self._show(finished, spoken_at=spoken_at, times=times, carried=carried)

    async def apply_heard(self, text: str) -> None:
        """A fragment of what the speaker said, in the source language (input_transcription)."""
        print(f"IN: {text}", flush=True)
        source, _target = languages_for(self._mode)
        heard_at = self._clock()
        self._heard.append((heard_at, text))
        await self._bus.publish(SpeechHeardEvent(text=text, language=source, heard_at=heard_at))

    async def apply_transcript(self, fragment: str, *, turn_complete: bool = False) -> None:
        if not self._active:
            return
        if not self._line.text:
            self._whole_sentences = self._next_whole_sentences
            self._clean_up_wording = self._next_clean_up_wording
        # The first English fragment is the closest this socket gets to when the line was spoken.
        at = self._clock() if fragment.strip() else None
        shown = self._line.extend(fragment, at=at) if fragment.strip() else ""
        if self._whole_sentences:
            await self._cut_long_line()
        spoken_at = self._line.opened_at
        text = self._finish(self._line.text) or ""
        if (turn_complete or _ends_sentence(text)) and not self._holds_short_sentence(text):
            self._stop_quiet_wait()
            times, carried = self._commit_line(text)
            if text:
                await self._show(text, spoken_at=spoken_at, times=times, carried=carried)
            return
        if self._whole_sentences:
            if self._line.text:
                self._commit_when_quiet()
        elif shown and self._in_order is None:
            # A line still being cleaned shows first; the next fragment shows this one grown.
            await self._publish(shown, provisional=True, spoken_at=spoken_at)

    def _commit_when_quiet(self) -> None:
        """Each fragment restarts the wait; a held line shows once the speaker pauses."""
        self._stop_quiet_wait()
        self._quiet = asyncio.create_task(self._commit_after_quiet(), name="gemini-live-quiet")

    def _stop_quiet_wait(self) -> None:
        """A committed or cleared line takes its quiet wait with it, so it can't cut the next line."""
        if self._quiet is not None:
            self._quiet.cancel()
            self._quiet = None

    async def _commit_after_quiet(self) -> None:
        # A line of only held words waits longer: it shows alone only if the speaker stopped.
        held_only = self._line.carried >= len(self._line.text.split())
        await asyncio.sleep(self._held_quiet_s if held_only else self._quiet_s)
        self._quiet = None
        await self._close_open_line()

    def _commit_line(self, text: str) -> tuple[list[float | None], int]:
        """Clear the open line, shown as `text` (its tidied words): when each of those words
        arrived, and how many of the first ones already waited from the caption before."""
        words = text.split()
        times = _times_of_kept(self._line.text.split(), self._line.word_times, words)
        carried = min(self._line.carried, len(words))
        self._line.commit()
        return times, carried

    def _finish(self, text: str | None) -> str | None:
        """The words a committed line shows: whole sentences drops fillers."""
        if text is None or not self._whole_sentences:
            return text
        return tidy(text) or None

    async def _cut_long_line(self) -> None:
        """Publish the front of a line that would overflow the overlay; the rest opens the next line."""
        while len(text := tidy(self._line.text)) > LONG_LINE_CHARS:
            cut = _cut_point(text, LONG_LINE_CHARS)
            if cut is None:
                return
            spoken_at = self._line.opened_at
            words = text.split()
            front = len(text[:cut].split())
            times, carried = self._commit_line(text)
            await self._show(
                text[:cut].strip(), spoken_at=spoken_at, times=times[:front], carried=min(carried, front)
            )
            if not self._active:
                # Captions went off while the front was shown; the off switch already cleared the line.
                return
            # The rest may have started in an earlier fragment; it keeps the times its words arrived.
            for word, word_at in zip(words[front:], times[front:]):
                self._line.extend(word, at=word_at)
            # Held words past the cut have waited once already, like the ones on the front.
            self._line.carried = max(0, carried - front)

    def _holds_short_sentence(self, text: str) -> bool:
        """A sentence like "Yes." waits to join the next one instead of filling a line alone."""
        return self._whole_sentences and len(text) < SHORT_SENTENCE_CHARS

    async def _show(
        self,
        text: str,
        *,
        spoken_at: float | None,
        times: list[float | None] | None = None,
        carried: int = 0,
    ) -> None:
        """A committed line, rewritten by the text model when the booth asks for clean wording."""
        # Decided now: a line still being cleaned keeps it if Whole sentences changes meanwhile.
        paced = self._whole_sentences
        cleaning = paced and self._clean_up_wording
        if cleaning and self._cleanup is None:
            logger.warning("cleanup skipped: no Gemini key")
            cleaning = False
        if not cleaning and self._in_order is None:
            await self._publish(text, provisional=False, spoken_at=spoken_at, paced=paced)
            self._shown.append(text)
            return
        # The socket keeps reading while the model works; lines still show in spoken order.
        self._waiting.append(
            _Waiting(
                text=text,
                times=times if times is not None else [spoken_at] * len(text.split()),
                spoken_at=spoken_at,
                committed_at=self._clock(),
                mode=self._mode,
                paced=paced,
                cleaning=cleaning,
                carried=carried,
            )
        )
        if self._in_order is None:
            self._in_order = asyncio.create_task(self._show_in_order(), name="gemini-live-in-order")

    def _drop_waiting_lines(self) -> None:
        """Captions off: lines still being cleaned never show, and their model calls stop."""
        for waiting in self._waiting:
            if waiting.call is not None:
                waiting.call.cancel()
        self._waiting.clear()

    async def _show_in_order(self) -> None:
        while self._waiting:
            entry = self._waiting[0]
            if entry.cleaning:
                start = entry.spoken_at if entry.spoken_at is not None else entry.committed_at
                said = self._said_between(start - SOURCE_LEAD_S, entry.committed_at)
                entry.call = asyncio.create_task(
                    self._cleaned(entry.text, said, hold=entry.carried == 0), name="gemini-live-cleanup"
                )
                await asyncio.wait({entry.call})
                if not self._waiting or self._waiting[0] is not entry:
                    # Captions went off while it was cleaned; lines after it are newer.
                    continue
                cleaned = entry.call.result()
            else:
                cleaned = Cleaned(entry.text)
            self._waiting.popleft()
            text = self._hold_unfinished(entry, cleaned)
            if not text:
                continue
            await self._publish(
                text,
                provisional=False,
                spoken_at=entry.spoken_at,
                mode=entry.mode,
                paced=entry.paced,
                before_cleanup=entry.text if entry.cleaning else None,
            )
            self._shown.append(text)
        self._in_order = None

    def _hold_unfinished(self, entry: _Waiting, cleaned: Cleaned) -> str:
        """What the line shows. Its unfinished words start the next line instead, when they can:
        each word waits only once, and only for a line of the same mode that is cleaned too."""
        if not cleaned.unfinished:
            return cleaned.finished
        held = cleaned.unfinished.split()
        everything = f"{cleaned.finished} {cleaned.unfinished}".strip()
        if len(held) > len(entry.text.split()) - entry.carried:
            return everything
        times = entry.times[-len(held):]
        if self._waiting:
            following = self._waiting[0]
            if following.mode is not entry.mode or not following.cleaning:
                return everything
            following.text = f"{cleaned.unfinished} {following.text}"
            following.times = times + following.times
            following.spoken_at = times[0] if times[0] is not None else following.spoken_at
            following.carried += len(held)
            return cleaned.finished
        if not self._active or self._mode is not entry.mode:
            return everything
        if not self._line.text:
            # The held words open the next line, which takes the booth's settings now.
            self._whole_sentences = self._next_whole_sentences
            self._clean_up_wording = self._next_clean_up_wording
            if not (self._whole_sentences and self._clean_up_wording):
                return everything
            self._line.prepend(cleaned.unfinished, times)
            # Shown when the speaker pauses, if no more words come to finish the sentence.
            self._commit_when_quiet()
        else:
            # The open line keeps the settings it started with; it may never be cleaned or wait.
            if not (self._whole_sentences and self._clean_up_wording):
                return everything
            self._line.prepend(cleaned.unfinished, times)
        return cleaned.finished

    def _said_between(self, since: float, until: float) -> str:
        """The speaker's own words heard from `since` to `until`, as one line."""
        return " ".join("".join(text for at, text in self._heard if since <= at <= until).split())

    async def _cleaned(self, text: str, said: str, *, hold: bool = True) -> Cleaned:
        """The text model's wording of a line, or the line as heard when the model can't answer.
        Without `hold`, the line shows whole: nothing of it waits for the next line."""
        assert self._cleanup is not None
        shown = list(self._shown)
        try:
            call = self._cleanup(text, shown, said) if hold else self._cleanup(text, shown, said, hold=False)
            reply = await asyncio.wait_for(call, timeout=self._cleanup_timeout_s)
            cleaned = reply if isinstance(reply, Cleaned) else Cleaned(reply)
            # What the model sent, before any check, so a skipped or odd line can be read back.
            logger.info(
                "cleanup reply: %s",
                json.dumps(
                    {"line": text, "finished": cleaned.finished, "unfinished": cleaned.unfinished},
                    ensure_ascii=False,
                ),
            )
            finished = cleaned.finished.strip()
            # What shows is "finished" (the unfinished words are checked as a copy below), so its
            # language is the one to match. Seen on the real model: an English caption came back
            # as the speaker's Korean.
            if finished and is_korean(finished) != is_korean(text):
                logger.warning("cleanup skipped: reply is in another language")
                return Cleaned(text)
            held = cleaned.unfinished.split()
            words = text.split()
            if held and not _copies_end_of(words, held):
                logger.warning("cleanup skipped: unfinished part is not the end of the line")
                return Cleaned(text)
            if held and not finished and len(held) < len(words):
                logger.warning("cleanup skipped: the reply leaves out words before the unfinished part")
                return Cleaned(text)
            answer = f"{finished} {cleaned.unfinished}".strip()
            if answer and 2 * len(answer) < len(text):
                # Cleanup trims fillers and false starts, not half a line: the rest was dropped.
                logger.warning("cleanup skipped: the reply is under half the line")
                return Cleaned(text)
            if held and not hold:
                # Held anyway: it still shows now, in the model's words. A cleaned "finished" wins
                # over a copy of the whole line, and a copy of its tail shows only once.
                if len(held) == len(words) or _copies_end_of(finished.split(), held):
                    return Cleaned(finished or " ".join(held))
                return Cleaned(f"{finished} {' '.join(held)}")
            # The line's own words wait, not the model's copy of them.
            unfinished = " ".join(words[len(words) - len(held) :]) if held else ""
            if held and len(held) == len(words):
                # Nothing of the line is finished, so a finished part could only be made up.
                return Cleaned("", unfinished)
            if finished or unfinished:
                return Cleaned(finished, unfinished)
            logger.warning("cleanup skipped: empty reply")
        except asyncio.TimeoutError:
            logger.warning("cleanup skipped: no reply within %s s", self._cleanup_timeout_s)
        except Exception as exc:
            logger.warning("cleanup skipped: %s", _redact(exc, self._api_key))
        return Cleaned(text)

    async def _publish(
        self,
        text: str,
        *,
        provisional: bool,
        spoken_at: float | None = None,
        mode: TranslateMode | None = None,
        paced: bool = False,
        before_cleanup: str | None = None,
    ) -> None:
        if not self._active and provisional:
            return
        # A line that waited on cleanup keeps the mode it was spoken in.
        mode = mode or self._mode
        source, target = languages_for(mode)
        await self._bus.publish(
            SubtitleEvent(
                chunk_id=str(uuid.uuid4()),
                text=text,
                source_language=source,
                target_language=target,
                mode=mode.value,
                created_at=self._clock(),
                duration_s=0.0,
                provisional=provisional,
                spoken_at=spoken_at,
                paced=paced,
                before_cleanup=before_cleanup,
            )
        )

    async def run(self) -> None:
        while True:
            if not self._active:
                await self._resumed.wait()
                continue
            self._reconnect.clear()
            try:
                async with self._connect(self._mode) as session:
                    await self._pump(session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("gemini live session failed: %s", _redact(exc, self._api_key))
                await self._close_open_line()
                await asyncio.sleep(1)
                continue
            await self._close_open_line()
            if not self._active:
                continue

    async def _pump(self, session: LiveConnection) -> None:
        sender = asyncio.create_task(self._send(session), name="gemini-live-send")
        receiver = asyncio.create_task(self._receive(session), name="gemini-live-recv")
        stop = asyncio.create_task(self._stop.wait(), name="gemini-live-stop")
        reconnect = asyncio.create_task(self._reconnect.wait(), name="gemini-live-reconnect")
        tasks = (sender, receiver, stop, reconnect)
        try:
            await asyncio.wait(set(tasks), return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def _send(self, session: LiveConnection) -> None:
        while self._active and not self._stop.is_set():
            try:
                pcm = await asyncio.wait_for(self._audio.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            await session.send_realtime_input(
                audio=_pcm_blob(pcm),
            )

    async def _receive(self, session: LiveConnection) -> None:
        async for response in session.receive():
            content = getattr(response, "server_content", None)
            if content is not None:
                heard = getattr(content, "input_transcription", None)
                heard_text = getattr(heard, "text", None) if heard is not None else None
                if heard_text:
                    await self.apply_heard(heard_text)
                transcription = getattr(content, "output_transcription", None)
                fragment = getattr(transcription, "text", None) if transcription is not None else None
                if fragment:
                    print(f"OUT: {fragment}", flush=True)
                turn_complete = bool(getattr(content, "turn_complete", False))
                if fragment or turn_complete:
                    await self.apply_transcript(fragment or "", turn_complete=turn_complete)
            if getattr(response, "go_away", None) is not None or not self._active:
                return


def _ends_sentence(text: str) -> bool:
    return bool(text) and text[-1] in ".?!"


@dataclass(slots=True)
class _Waiting:
    """A committed line waiting to show: its words, when each arrived, when it was spoken and
    committed, its mode, whether it is paced and cleaned, how many of its first words waited
    from the line before, and its cleanup call once started."""

    text: str
    times: list[float | None]
    spoken_at: float | None
    committed_at: float
    mode: TranslateMode
    paced: bool
    cleaning: bool
    carried: int = 0
    call: asyncio.Task[Cleaned] | None = None


def is_korean(text: str) -> bool:
    """Whether a caption is Korean: more Hangul than Latin letters. An English caption that quotes
    a Korean word ("the love of 하나님") is still English; a Korean one naming "Romans" is Korean."""
    hangul = sum("가" <= ch <= "힣" for ch in text)
    latin = sum(ch.isascii() and ch.isalpha() for ch in text)
    return hangul > latin


def _ends_with_words(words: list[str], tail: list[str]) -> bool:
    """Whether `tail` is the last words of `words`, whatever punctuation either copy has."""
    if not 0 < len(tail) <= len(words):
        return False
    return [_bare(w) for w in words[-len(tail) :]] == [_bare(w) for w in tail]


def _copies_end_of(words: list[str], tail: list[str]) -> bool:
    """Whether `tail` is the model's copy of the last words of `words`. The Korean style
    rewords "당신은" to "여러분은" even in a copy, so that one change, in that direction only,
    still counts as a copy."""
    if not 0 < len(tail) <= len(words):
        return False
    return all(
        _bare(copy) in (_bare(word), _as_styled(word)) for word, copy in zip(words[-len(tail) :], tail)
    )


# The Korean style's "never 당신 or 너": how a copy of the line may have rewritten them.
_YOU_AS_STYLED = {"너는": "여러분은", "너를": "여러분을", "너의": "여러분의", "네가": "여러분이", "너에게": "여러분에게", "너": "여러분"}


def _as_styled(word: str) -> str:
    word = _bare(word)
    if word.startswith("당신"):
        return "여러분" + word[len("당신") :]
    return _YOU_AS_STYLED.get(word, word)


def _bare(word: str) -> str:
    return word.strip(",;:.?!—–-…\"'").lower()


def _times_of_kept(raw: list[str], times: list[float | None], kept: list[str]) -> list[float | None]:
    """When each tidied word arrived. Tidying only drops words, so kept words match raw ones in order."""
    out: list[float | None] = []
    i = 0
    for word in kept:
        while i < len(raw) and _bare(raw[i]) != _bare(word):
            i += 1
        if i < len(raw):
            out.append(times[i])
            i += 1
        else:
            out.append(times[-1] if times else None)
    return out


def _cut_point(text: str, limit: int) -> int | None:
    """Where to end a line of at most `limit` characters: its last comma, else its last space.

    A comma in the first half would leave a stub of a line, so the space wins there.
    """
    comma = text.rfind(",", 0, limit)
    if comma >= limit // 2:
        return comma + 1
    space = text.rfind(" ", 0, limit + 1)
    return space if space > 0 else None


def _pcm_blob(pcm: bytes) -> object:
    from google.genai import types

    return types.Blob(data=pcm, mime_type="audio/pcm;rate=16000")


def _redact(exc: BaseException, api_key: str) -> str:
    text = str(exc)
    if api_key:
        text = text.replace(api_key, "[key]")
    return text
