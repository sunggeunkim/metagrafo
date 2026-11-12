# Architecture

Metagrafo is a single-process, fully local speech pipeline that turns live **program audio** into OBS captions.

OBS on the broadcast PC is the encoder (YouTube and the house projector share that canvas). Mixer audio reaches OBS over Stream to USB RTMP; Metagrafo captures **`HDMI (Elgato 4K S)`**. Silero VAD chunks speech; faster-whisper produces English; a FastAPI WebSocket overlay shows **two completed lines**. Hardware path: `docs/aten-obs.md`. GPU/OS profiles: `docs/hardware-profiles.md`.

## Product locks (v1)

| Topic | Lock |
|---|---|
| Encoder | OBS encodes the destination. UC9020 RTMP alone has **no captions**. |
| Audio | PGM mix. No speech-only bus. |
| Worship | Captions **default ON**. Volunteer OFF for songs. Mute **stops inference**, not only the OBS source. |
| Presentation | Two-line. Whisper is utterance-final (**not karaoke**). Gemini live, when opted in, shows finished lines too (**Whole sentences**, default on); with that off it grows the bottom line while the pastor is still talking, then moves that line up. ~2–3 s after they stop is OK for Whisper. |
| Modes | `ko_to_en` (sermon), `en_to_en` (English guest transcribe), and `en_to_ko` (English guest, Korean captions; **Gemini live only**). NLLB deferred. |
| Code-switch | Stay on `ko_to_en`. No verse-by-verse mode flips. |
| Mute in-flight | Capture drops frames. Translate empties the queue. In-flight Whisper may finish; **discard if captions are off**. |
| Model | Env `WHISPER_MODEL` (default `large-v3`). Fallback `medium` + **restart**. No live swap. |

## Vertical Slice Architecture

Organize by **feature**, not by layer.

Each slice owns its events, I/O, and runtime loop. Slices talk only through `core/event_bus.py`. Downstream slices may import **events** from upstream. They must not import each other's internals.

**Do not create** `controllers/`, `services/`, `repositories/`, `models/`, `domain/`, `infrastructure/`, `api/`, or `schemas/` as shared layers.

Allowed shared code:

- `core/event_bus.py` — typed asyncio pub/sub (no event types in this file)
- `core/settings.py` — env-backed bootstrap; hardware profile fills **unset** fields only
- `main.py` — composition root

```
PGM audio → capture_audio --AudioChunkEvent--> translate_speech --SubtitleEvent--> broadcast_subtitles → OBS
                                                                              ├-----> youtube_captions → YouTube CC (optional)
                                                                              └-----> record_transcript → transcripts/*.jsonl (optional)

operator_control --CaptionsStateEvent / ModeChangedEvent-->  (the three slices above subscribe)
```

`operator_control` publishes reverse control. Capture does **not** import the overlay.

## Layout

```
metagrafo/
  main.py
  core/
    event_bus.py
    settings.py
  features/
    capture_audio/          # named input → AudioChunkEvent
    translate_speech/       # AudioChunkEvent → SubtitleEvent (no HTTP)
    broadcast_subtitles/    # /overlay, /ws, /health
    operator_control/       # /control, /mode, CaptionsStateEvent
    youtube_captions/       # SubtitleEvent → YouTube caption ingestion URL, /youtube-captions
    record_transcript/      # SubtitleEvent + SpeechHeardEvent → transcripts/*.jsonl, /transcript
  docs/
    architecture.md
    aten-obs.md
    hardware-profiles.md
```

Each feature exposes `register(...)`. FastAPI routes live in the slice that owns that surface.

## Feature slices

### `capture_audio`

Opens the **named** recording device (Windows default `HDMI (Elgato 4K S)`). Among name matches, prefer a **2-channel** endpoint (WASAPI over 16-channel MME). Overridable by name/index (`AUDIO_DEVICE_NAME` / `AUDIO_DEVICE_INDEX`). Subscribe `AudioDeviceChangedEvent`: stop the capture thread and reopen with the name now on `Settings` (live, no restart). `status()` reports the device it opened, or why the open failed; `main.py` hands it to `operator_control`. PortAudio initialize/terminate are not thread-safe, so every PyAudio create/open/close/terminate holds `aten_mic.PORTAUDIO_LOCK`.

- Windows: shared WASAPI (not exclusive) so OBS can use the same device.
- Downmix stereo → mono, resample to 16 kHz, 512-sample frames.
- Silero VAD (ONNX): ~200 ms pre-roll, ~500 ms trailing silence, ~12 s cap, drop &lt;~250 ms. Pauses per caption default **1** (one line per pause). **2** waits for a second pause, then prints each phrase on its own line. The 12 s cap still cuts.
- Subscribe `CaptionsStateEvent`: if inactive, **drop frames, no VAD**.
- PCM on the bus is `bytes` (s16le).
- No FastAPI routes.

### `translate_speech`

Subscribes to `AudioChunkEvent`, `CaptionsStateEvent`, and mode. No `/control`.

| Mode | Whisper language | Whisper task | On screen |
|---|---|---|---|
| `ko_to_en` | `ko` | `translate` | English |
| `en_to_en` | `en` | `transcribe` | English (unchanged) |

`en_to_ko` has no Whisper row: Whisper only translates into English. Gemini live opens its socket with target `ko`. `PUT /mode` answers **409** for `en_to_ko` until a Gemini key is saved, and a Whisper worker started with `TRANSLATE_MODE=en_to_ko` falls back to `ko_to_en`.

- `large-v3` (`turbo` ignores `task="translate"`). Inference in `asyncio.to_thread`.
- `HERMENEIA_URL` sends each VAD chunk to that box (`/v1/translations`) instead of local Whisper. `HERMENEIA_TOKEN` is the bearer token. An unset URL keeps local Whisper. Gemini live does not call Hermeneia.
- `--compare` (prototype for #41, `compare_run.py`): one real-time Gemini live pass over a wav writes `<name>.compare.md`. B feeds the English, and the Korean heard, into a private `GeminiLiveTranslator` (Whole sentences and Clean up wording on), so its cleanup sees what the speaker said as live cleanup does. C closes each Korean `input_transcription` sentence at `.?!` or a `WHOLE_SENTENCES_QUIET_S` pause and translates it in one `CLEANUP_MODEL` call with the last 3 C lines as context, ignoring Korean fillers. Reconnects on `go_away`. Not wired into the live pipeline or booth control.
- `condition_on_previous_text` defaults **on** (the caption just written is passed into the next phrase). **Off** translates each phrase alone. Live from `/control`. A file run turns it off with `--no-condition-on-previous-text`.
- Device profile: see `docs/hardware-profiles.md`. Env always wins.
- Queue max 4, **drop oldest**. Empty output is not published.
- Captions off: **empty the queue**. When Whisper returns, if inactive, **do not publish**.
- `CAPTION_ENGINE=gemini_live` plus `GEMINI_API_KEY` hears program audio as it arrives (`ProgramAudioEvent`, about 100 ms) instead of waiting for a VAD chunk. `gemini-3.5-live-translate-preview` translates Korean speech to English on one socket, the same four settings as the working mic test: no context-window compression and no session resumption. The bottom line grows (`SubtitleEvent.provisional`), and `spoken_at` is when its first English fragment arrived; a finished sentence, a model turn, or the end of a socket commits it. **Whole sentences** (`WHOLE_SENTENCES`, default **on**) publishes committed lines only, never `provisional`: a finished sentence under 25 characters waits and joins the next one, a line past 120 characters (two default overlay rows hold about 140; slack until #36) is cut after its last comma in the second half, else at its last space, and ~1.2 s with no new fragment (`WHOLE_SENTENCES_QUIET_S`) commits the open line. Committed lines drop pure fillers (uh, um, er, hmm) and immediate stutters ("the the", "we— we"); "you know" and "I mean" stay. **Clean up wording** (`CLEAN_UP_WORDING`, default **on**; needs Whole sentences) sends each committed line, with the last 3 shown lines as context and what the speaker said around it (the `input_transcription` heard from `SOURCE_LEAD_S`, 6 s, before the line's first translated fragment until it is committed, so it may include a little of the lines either side), to a Gemini text model (`CLEANUP_MODEL`, default `gemini-3.5-flash-lite`) on the saved Gemini key, asking it to drop disfluencies and false starts, lightly fix grammar, keep the meaning, add nothing, and keep scripture references and names as spoken. A Korean line is also worded in the booth's **Korean style** (`korean-style.txt` beside `.env`, up to 1000 characters; no file means the default, formal sermon language: 합니다체, never 해요체 or 반말; honorifics for God and Jesus; the congregation is 여러분, never 당신 or 너; natural Korean word order, folding trailing time, place and duration fragments into their sentence, which works only within one cleaned line; an empty file means no style rule), as part of cleaning it. The booth's **Glossary** (`glossary.txt`, up to 3000 characters; no file means `GLOSSARY_DEFAULT`, an empty file means none) is appended too, for church terms in either direction. With what the speaker said beside the line, the model corrects mistranslated words ("동성애 문제" for "the issue of suffering") instead of keeping or dropping them. **Additional context** (booth control, kept only while Metagrafo runs, up to 2000 characters) is appended to that instruction: the model follows any wording or style instructions in it and uses the rest as background for names, scripture and terms, never to be added to a line; the cleanup reads it for every line, so a change applies from the next one. The live translate model itself takes no instruction (Google lists system instructions as not supported). A line is cut wherever the translation pauses, so it can end mid-sentence. The model replies in JSON (Gemini structured output, `REPLY_SCHEMA`): `finished`, the cleaned complete sentences, and `unfinished`, the words at the end of the line that don't finish a sentence yet, copied from the line and never completed from the speaker's words. Only `finished` shows. The line's own `unfinished` words (matched to the model's copy ignoring punctuation) start the next line with the times they arrived: the next committed line if one is waiting (same mode, also cleaned), else the open line. A line of only held words waits `HELD_QUIET_S` (5 s) instead of the usual quiet for more words, since the translation often pauses longer than 1.2 s mid-sentence. Each word is held at most once: a line that starts with held words is cleaned with `hold=False`, which tells the model the line shows now, so it puts everything in `finished`, stops where the line stops (never adding words to finish it) and still follows the Korean style; if the model holds some anyway, its own words for them show after `finished`. A copy whose only change is the Korean style's 당신/너 → 여러분 still counts as the end of the line. A reply with an empty `finished` and only part of the line in `unfinished` would lose words, so the line shows as heard (`cleanup skipped: the reply leaves out words before the unfinished part`). An `unfinished` that isn't the end of the line shows the line as heard (`cleanup skipped: unfinished part is not the end of the line`); when `unfinished` is the whole line, a `finished` part could only be made up and is dropped. Lines are therefore cleaned one at a time, in order, so each sees the one before it among the shown lines. The socket keeps reading meanwhile; lines still show in spoken order (a growing line waits behind one being cleaned) and keep the mode they were spoken in. No reply within 4 s (`CLEANUP_TIMEOUT_S`), an error, a reply that isn't that JSON, an empty reply, or no key shows the line as heard and logs `cleanup skipped: <reason>` (never the key). Every reply is logged before those checks, one JSON line each: `cleanup reply: {"line", "finished", "unfinished"}`, Korean as written; a reply that isn't JSON is skipped as `reply is not JSON: '<first 300 characters>'`. The cleaned line is what the overlay and YouTube CC get. Captions off drops lines still being cleaned and cancels their calls. The client pins API `v1beta`, like the socket, so google-genai 2.x and 3.x send the same request. Off is the growing line above. A switch takes effect at the next line; the open line finishes the way it started. Whisper ignores it. Captions off clears the open line and the overlay. English guest mode reconnects with echo so English speech is captioned. The microphone match prefers the WASAPI device, because the DirectSound copy of the same name does not wait for the clock. Missing key keeps Whisper. The older `Captioner` engine named `gemini` is not this path.

### `broadcast_subtitles`

- `WS /ws` — caption and mode JSON. On connect it sends the overlay style, then the current `captions_state`, so an overlay that reconnects after Stop captions clears its old lines.
- `GET /overlay` — OBS Browser Source (no booth buttons)
- `GET /health` — liveness, client count, captions active, mode, model/device/VRAM

**Two lines at a time** (`ONE_SENTENCE`, default **on**; the setting kept its first name): the overlay shows whole lines, up to **2** rows on screen. A finished line (or the growing one) joins the lines on screen while they all fit in 2 rows; otherwise the screen clears and it starts afresh. A growing line that clears the screen keeps it cleared, so its finished version also starts afresh. A line longer than 2 rows shows whole, alone; `line_count` does not cut it. A line taller than the box loses its first rows instead of being clipped at the bottom, so its end stays readable. Off: rolling rows, default **3** (newest at the bottom, older lines above), and `/control` sets **1–6** lines. Designed for a **1920×1080** OBS Browser Source (do not scale the source — that squashes type). Caption box width/height are percent of that overlay (defaults **100 × 30**). Wrap on word boundaries; Korean breaks only at spaces. Rows are found by measuring each character's own glyph rect in a hidden copy; empty rects are ignored, because OBS's Chromium 150 also reported one at the end of the row above for the first character of a row, which split a word across two boxes. Each box shows exactly one row, so the line count is the row count. New `text` shifts older lines up. Stop captions **clears the lines**. Otherwise lines stay up until newer lines push them out; there is no quiet fade, so a long sentence still being spoken never leaves the screen blank.

- Pacing (`/ws` only; YouTube CC is not paced). A `SubtitleEvent.paced` line stays up at least `max(2.5 s, characters / 9 per second)` before the next paced line is sent, and its message carries that time as `min_on_screen_s` (the overlay itself no longer fades, so it ignores it). Paced lines that arrive sooner wait in order. If the newest waiting line would show more than ~5 s after it arrived, the **oldest waiting line is dropped** (repeat); the newest is never dropped. Dropping, not merging: a merged line takes as long to read as its parts. Unpaced lines (Whisper, `provisional`, Whole sentences off, `/inject`) go out at once, past any waiting lines, and do not restart the wait. Stop captions empties the waiting lines and cancels the pending wait, so none is sent later; the next paced line shows at once. `SubtitleHub(min_on_screen_s, chars_per_s, max_behind_s)`.

```json
{
  "type": "subtitle",
  "text": "Hello, everyone.",
  "chunk_id": "...",
  "source_language": "ko",
  "target_language": "en",
  "mode": "ko_to_en",
  "ts": 1720000000.12,
  "duration_s": 2.4
}
```

Browser tests render `/overlay` in headless Chromium at 1280 and 1920 px. Run `uv run playwright install chromium` once; without it they skip.

### `youtube_captions`

Optional. `YOUTUBE_CAPTION_URL` is the **Captions Ingestion URL** from YouTube Studio (**POST captions to URL**). Unset sends nothing. `/control` can save or clear it live.

- Subscribes to `SubtitleEvent`. Skips `provisional` lines and blank text.
- One line per POST to `<url>&seq=N`, `Content-Type: text/plain`, body `YYYY-MM-DDTHH:MM:SS.mmm\n<text>\n` in UTC. The stamp is `SubtitleEvent.spoken_at` (wall clock when the speech began: the VAD chunk start for Whisper, the line's first English fragment for Gemini live), or `created_at` when that is unset.
- One sender task posts in order, so `seq` only rises. `seq` advances on a 2xx. A refused line (4xx/5xx) is dropped and its `seq` is reused. A timeout or broken connection retries the same line and `seq` twice (0.5 s, 1 s); if it still fails the line is dropped and `seq` moves on, because YouTube may already have it.
- Queue max 16, **drop oldest**.
- YouTube refuses lines until the stream is live: warn once, log once when it recovers. The URL is never logged (it carries the caption key); a filter on the `httpx` logger drops its INFO request lines for this URL.
- `YouTubeCaptionUrlChangedEvent` — read the new URL from Settings. Empties the queue and starts `seq` and the counts over at 1/0 (a new URL is a new stream). An empty URL drops lines. A line still being sent when the URL changes is abandoned, never sent to the new URL.
- `GET /youtube-captions` — `enabled`, `sent`, `failed`, `last_status` (only `enabled: false` without a URL).

### `record_transcript`

For reading the captions back afterwards (analysis), not for viewers. Off by default.

- `SaveTranscriptEvent(save_transcript)` — on opens a new `transcripts/YYYY-MM-DD_HHMMSS.jsonl` (in the folder Metagrafo starts in; `_2`, `_3`… when that name is taken, so a session never appends to another); off closes it. `SAVE_TRANSCRIPT=true` in `.env` opens one at startup. Shutdown, including **Quit Metagrafo**, closes it. Lines are flushed as written, so a file can be read mid-service.
- One JSON record per line. Every record has `t_s` (seconds since this file started), `at` (local clock, ms) and `kind`:
  - `start` — `engine` (`gemini_live` \| `whisper`), `captions_active`, `mode`, `whole_sentences`, `clean_up_wording`, `cleanup_context`, `korean_style`, `glossary`.
  - `heard` — `text`, `language`: a Korean (source) fragment as Gemini live transcribed it (`SpeechHeardEvent`). Whisper writes none.
  - `shown` — `text` as the overlay and YouTube got it, `delay_s` (shown time − `spoken_at`; for Gemini live that is from the line's first English fragment, so a lower bound on the delay from speech; compare with the `heard` times), `mode`, and `before_cleanup` when Clean up wording rewrote it (equal to `text` when the model kept it or timed out; it starts with any words the line before held back, and `text` leaves out the words this line held back for the next). Provisional (growing) lines are skipped.
  - `setting` — one booth change: `captions_active`, `mode`, `whole_sentences`, `clean_up_wording`, `cleanup_context`, `korean_style`, or `glossary`.
- `GET /transcript` — `{saving, file}` with the file's full path.

### `operator_control`

Owns the booth UI. Publishes only.

- `GET /control` — Start/Stop captions (one button that flips), Show/Hide overlay and Show/Hide logs (both open when the page loads, side by side under the top buttons: a scaled 1920×1080 `/overlay` frame on the left that connects only while shown, and a log panel on the right that polls `/logs` every 2 s while open), Mode ko_to_en, Mode en_to_en (Guest), Mode en_to_ko (Guest → Korean; disabled until Gemini live), **VAD pause (ms)**, **pauses per caption (1 or 2)**, **previous text** (`condition_on_previous_text`, default on), **Whole sentences** checkbox (Gemini live, default on), **Clean up wording** checkbox (default on; greyed out while Whole sentences is off), **Additional context** box (Save context / Clear), **Korean style** box (Save style / Reset to default), **Glossary** box (Save glossary / Reset to default), **Save transcript** checkbox (default off; shows the file being written), overlay position, **font size (vw)**, **box width/height (%)**, **edge insets (%)**, **Two lines at a time** checkbox (`ONE_SENTENCE`, default on), **lines on screen (1–6)** (greyed out while Two lines at a time is on), Gemini key, **YouTube caption URL** (Save URL / Stop sending, plus sent/failed counts), **audio input** (pick a listed recording device), **Quit Metagrafo** (asks first, then replaces the page with how to start it again)
- `GET /mode` / `PUT /mode` / `PUT /vad-silence` / `PUT /vad-pauses` / `PUT /condition-on-previous-text` / `PUT /whole-sentences` / `PUT /one-sentence` / `PUT /clean-up-wording` / `PUT /overlay-style` / `PUT /gemini-key` / `PUT /youtube-caption-url` / `GET /audio-devices` / `PUT /audio-device`
- `GET /audio-devices` — `{devices, selected, capturing, error}`: input names this PC lists now (each once; names with a line break are left out because `.env` cannot hold them), `AUDIO_DEVICE_NAME`, and capture's status. `main.py` injects the lister and the status, so this slice does not import capture code.
- `PUT /audio-device` — `{name}`; must be one of the listed names (422 otherwise), and 409 while `AUDIO_DEVICE_INDEX` is set. Writes `AUDIO_DEVICE_NAME` to `.env`, keeping other lines, and publishes `AudioDeviceChangedEvent`. `/mode` shows `audio_device`, `audio_capturing`, and `audio_error`; the booth page polls those every 5 s.
- `PUT /cleanup-context` — `{text}` (at most 2000 characters, 422 otherwise; trimmed; empty clears). Kept in memory only, so every start is empty (a `cleanup-context.txt` left by an earlier version is ignored); sets `Settings.cleanup_context`, and publishes `CleanupContextEvent`. `/mode` shows `cleanup_context`; the page fills its box once on load.
- `PUT /korean-style` — `{text}` (at most 1000 characters, 422 otherwise; trimmed; empty turns the rule off and is kept). Writes `korean-style.txt` beside `.env`, sets `Settings.korean_style`, and publishes `KoreanStyleEvent`. `DELETE /korean-style` removes the file and goes back to `KOREAN_STYLE_DEFAULT`. `/mode` shows `korean_style` and `korean_style_default`.
- `PUT /glossary` / `DELETE /glossary` — the same for the church glossary: at most 3000 characters, `glossary.txt` beside `.env`, `Settings.glossary`, `GlossaryEvent`, default `GLOSSARY_DEFAULT` (general church vocabulary and 개역개정 Bible book names, never one sermon's terms). `/mode` shows `glossary` and `glossary_default`.
- `PUT /save-transcript` — `{save_transcript}`; writes `SAVE_TRANSCRIPT` to `.env`, keeping other lines, and publishes `SaveTranscriptEvent`. `/mode` and `/health` show `save_transcript`.
- `POST /quit` — `{quitting: true}`, then the server shuts down gracefully (`main.build_server` sets uvicorn's `should_exit`), so the lifespan still stops capture, the translator and YouTube sending; open sockets get 5 s to close. `main.py` injects `request_quit`; an app without it (`uvicorn main:app`, tests) answers 409.
- `GET /logs?after=<offset>` — `{available, text, offset}`: whole lines of `metagrafo.log` from byte `after` on (without it, the last ~250 KB), and the `offset` to pass next. A line still being written waits for the next call. Only a windowless start (`pythonw`, `Start Metagrafo.cmd`) writes that file, so `main.py` passes the path only then; a console start answers `available: false`.
- `PUT /youtube-caption-url` — `{url}`; empty clears. Must be http(s) with a host and no whitespace (422 otherwise). Writes `YOUTUBE_CAPTION_URL` to `.env`, keeping other lines, and publishes `YouTubeCaptionUrlChangedEvent`. The URL is never echoed back; `/mode` shows only `youtube_url_set`.
- `CaptionsStateEvent(is_active: bool)` — default **true** at process start
- `ModeChangedEvent` — pre-service; not verse-by-verse
- `VadSilenceMsEvent` — live; capture updates the running chunker (100–3000 ms)
- `VadPausesEvent` — live; 1 cuts on each pause. 2 waits for a second pause, then one line per phrase
- `ConditionOnPreviousTextEvent` — live; feed the caption just written into the next phrase
- `OneSentenceEvent` — live. `PUT /one-sentence` (`{one_sentence}`) writes `ONE_SENTENCE` to `.env`; the hub adds `one_sentence` to every `overlay_style` message (on connect and on change), so the overlay switches at once.
- `WholeSentencesEvent` — live; Gemini live shows only finished lines from the next line on. `PUT /whole-sentences` (`{whole_sentences}`) writes `WHOLE_SENTENCES` to `.env`, keeping other lines, so the choice survives a restart; `/mode` and `/health` show `whole_sentences`
- `CleanUpWordingEvent` — live; a text model rewrites each finished Gemini live line from the next line on. `PUT /clean-up-wording` (`{clean_up_wording}`) writes `CLEAN_UP_WORDING` to `.env`, keeping other lines; `/mode` and `/health` show `clean_up_wording`
- `OverlayStyleEvent` — live over `/ws` (`position` `top_left` \| `top_center` \| `bottom_left` \| `bottom_center`, `font_size_vw` 1–8, `box_width_pct` 10–100, `box_height_pct` 5–100, `inset_vertical_pct` / `inset_horizontal_pct` 0–20, `line_count` 1–6). `line_count` is visible text rows, default 3 (a ~100-character Korean line takes three rows at 2.5 vw). A wrapped sentence keeps the newest rows and the oldest row leaves.

Open `/control` in a normal browser, **not** as an OBS source.

## Event bus

In-process asyncio pub/sub. Dispatch by exact type. `publish` isolates handler exceptions. No singleton; `main.py` injects one bus.

Producer-owned events:

- `AudioChunkEvent` — `capture_audio`
- `SubtitleEvent` — `translate_speech` (read by `broadcast_subtitles`, `youtube_captions` and `record_transcript`). `paced` (default false) asks the overlay to keep the line up long enough to read; Gemini live sets it on lines committed while Whole sentences is on (a line still being cleaned keeps it if the switch changes), Whisper never does. `before_cleanup` is the line as heard when Clean up wording sent it to the text model, else None; only the transcript reads it
- `SpeechHeardEvent(text, language, heard_at)` — `translate_speech` (Gemini live `input_transcription`; read by `record_transcript`)
- `CaptionsStateEvent`, `ModeChangedEvent`, `VadSilenceMsEvent`, `VadPausesEvent`, `ConditionOnPreviousTextEvent`, `WholeSentencesEvent`, `CleanUpWordingEvent`, `OverlayStyleEvent`, `CleanupContextEvent`, `KoreanStyleEvent`, `GlossaryEvent`, `SaveTranscriptEvent`, `GeminiKeyChangedEvent`, `YouTubeCaptionUrlChangedEvent`, `AudioDeviceChangedEvent` — `operator_control`

## Process model

One process, one asyncio loop, uvicorn.

| Work | Where |
|---|---|
| PyAudio read | capture thread |
| Silero VAD | capture thread (only if captions on) |
| faster-whisper | `asyncio.to_thread` |
| WebSocket fan-out | asyncio |

Startup: operator_control + translate + broadcast + youtube_captions, then capture. Shutdown: stop capture first.

## Settings

Env (`core/settings.py`). Profile fills unset whisper fields only.

Notable: `AUDIO_DEVICE_NAME` (default `HDMI (Elgato 4K S)`), `AUDIO_DEVICE_INDEX`, VAD (`VAD_MIN_SILENCE_MS` default 500), `WHISPER_MODEL` (default `large-v3`), `WHISPER_DEVICE` (default `cuda`), `WHISPER_COMPUTE_TYPE` (default `float16`), `TRANSLATE_MODE` (`ko_to_en` \| `en_to_en` \| `en_to_ko`), queue size, host/port, `YOUTUBE_CAPTION_URL` (optional), `WHOLE_SENTENCES` (default `true`), `WHOLE_SENTENCES_QUIET_S` (default 1.2), `CLEAN_UP_WORDING` (default `true`), `CLEANUP_MODEL` (default `gemini-3.5-flash-lite`), `CLEANUP_TIMEOUT_S` (default 4), `SAVE_TRANSCRIPT` (default `false`), `ONE_SENTENCE` (default `true`). Captions active is **not** persisted; always starts on.

## Operator path

1. Church PC: **thewaychurch** sign-in runs `scripts/Metagrafo.cmd` in that user session (not SYSTEM). Captions **ON**, mode `ko_to_en`.
2. OBS: ATEN video + audio + Browser Source `/overlay`. House projector = program.
3. `/control` in a booth browser.
4. Pastor at pulpit → Start captions. Worship → Stop captions.
5. English guest: set **en_to_en** before service, or **en_to_ko** (Gemini only) to caption them in Korean.
6. Rehearsal hitch: `WHISPER_MODEL=medium`, restart. Never hot-swap on the GPU.

## Out of scope (v1)

- NLLB (local `en_to_ko` without Gemini)
- Karaoke / partials / word highlight
- Application database
- whisper.cpp Metal
- Live VRAM/model swap
- Speech-only aux mix
- Cloud ASR
