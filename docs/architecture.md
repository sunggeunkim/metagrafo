# Architecture

Metagrafo is a single-process, fully local speech pipeline that turns live **program audio** into OBS captions.

OBS on the broadcast PC is the encoder (YouTube and the house projector share that canvas). Mixer audio reaches OBS over Stream to USB RTMP; Metagrafo captures **`HDMI (Elgato 4K S)`**. Silero VAD chunks speech; faster-whisper produces English; a FastAPI WebSocket overlay shows **two completed lines**. Hardware path: `docs/aten-obs.md`. GPU/OS profiles: `docs/hardware-profiles.md`.

## Product locks (v1)

| Topic | Lock |
|---|---|
| Encoder | OBS encodes the destination. UC9020 RTMP alone has **no captions**. |
| Audio | PGM mix. No speech-only bus. |
| Worship | Captions **default ON**. Volunteer OFF for songs. Mute **stops inference**, not only the OBS source. |
| Presentation | Two-line. Whisper is utterance-final (**not karaoke**). Gemini live, when opted in, grows the bottom line while the pastor is still talking, then moves that line up. ~2–3 s after they stop is OK for Whisper. |
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
                                                                              └-----> youtube_captions → YouTube CC (optional)

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
  docs/
    architecture.md
    aten-obs.md
    hardware-profiles.md
```

Each feature exposes `register(...)`. FastAPI routes live in the slice that owns that surface.

## Feature slices

### `capture_audio`

Opens the **named** recording device (Windows default `HDMI (Elgato 4K S)`). Among name matches, prefer a **2-channel** endpoint (WASAPI over 16-channel MME). Overridable by name/index (`AUDIO_DEVICE_NAME` / `AUDIO_DEVICE_INDEX`).

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
- `condition_on_previous_text` defaults **on** (the caption just written is passed into the next phrase). **Off** translates each phrase alone. Live from `/control`. A file run turns it off with `--no-condition-on-previous-text`.
- Device profile: see `docs/hardware-profiles.md`. Env always wins.
- Queue max 4, **drop oldest**. Empty output is not published.
- Captions off: **empty the queue**. When Whisper returns, if inactive, **do not publish**.
- `CAPTION_ENGINE=gemini_live` plus `GEMINI_API_KEY` hears program audio as it arrives (`ProgramAudioEvent`, about 100 ms) instead of waiting for a VAD chunk. `gemini-3.5-live-translate-preview` translates Korean speech to English on one socket, the same four settings as the working mic test: no context-window compression and no session resumption. The bottom line grows (`SubtitleEvent.provisional`), and `spoken_at` is when its first English fragment arrived; a finished sentence, a model turn, or the end of a socket commits it. Captions off clears the open line and the overlay. English guest mode reconnects with echo so English speech is captioned. The microphone match prefers the WASAPI device, because the DirectSound copy of the same name does not wait for the clock. Missing key keeps Whisper. The older `Captioner` engine named `gemini` is not this path.

### `broadcast_subtitles`

- `WS /ws` — caption and mode JSON
- `GET /overlay` — OBS Browser Source (no booth buttons)
- `GET /health` — liveness, client count, captions active, mode, model/device/VRAM

Overlay lines: default **2** (newest at the bottom, older lines above). `/control` sets **1–6** lines. Designed for a **1920×1080** OBS Browser Source (do not scale the source — that squashes type). Caption box width/height are percent of that overlay (defaults **100 × 30**). Wrap on word boundaries. New `text` shifts older lines up. Captions OFF **clears the lines**. Fade after a quiet interval.

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

### `youtube_captions`

Optional. `YOUTUBE_CAPTION_URL` is the **Captions Ingestion URL** from YouTube Studio (**POST captions to URL**). Unset sends nothing. `/control` can save or clear it live.

- Subscribes to `SubtitleEvent`. Skips `provisional` lines and blank text.
- One line per POST to `<url>&seq=N`, `Content-Type: text/plain`, body `YYYY-MM-DDTHH:MM:SS.mmm\n<text>\n` in UTC. The stamp is `SubtitleEvent.spoken_at` (wall clock when the speech began: the VAD chunk start for Whisper, the line's first English fragment for Gemini live), or `created_at` when that is unset.
- One sender task posts in order, so `seq` only rises. `seq` advances on a 2xx. A refused line (4xx/5xx) is dropped and its `seq` is reused. A timeout or broken connection retries the same line and `seq` twice (0.5 s, 1 s); if it still fails the line is dropped and `seq` moves on, because YouTube may already have it.
- Queue max 16, **drop oldest**.
- YouTube refuses lines until the stream is live: warn once, log once when it recovers. The URL is never logged (it carries the caption key); a filter on the `httpx` logger drops its INFO request lines for this URL.
- `YouTubeCaptionUrlChangedEvent` — read the new URL from Settings. Empties the queue and starts `seq` and the counts over at 1/0 (a new URL is a new stream). An empty URL drops lines. A line still being sent when the URL changes is abandoned, never sent to the new URL.
- `GET /youtube-captions` — `enabled`, `sent`, `failed`, `last_status` (only `enabled: false` without a URL).

### `operator_control`

Owns the booth UI. Publishes only.

- `GET /control` — Captions ON/OFF, Mode ko_to_en, Mode en_to_en (Guest), Mode en_to_ko (Guest → Korean; disabled until Gemini live), **VAD pause (ms)**, **pauses per caption (1 or 2)**, **previous text** (`condition_on_previous_text`, default on), overlay position, **font size (vw)**, **box width/height (%)**, **edge insets (%)**, **lines on screen (1–6)**, Gemini key, **YouTube caption URL** (Save URL / Stop sending, plus sent/failed counts)
- `GET /mode` / `PUT /mode` / `PUT /vad-silence` / `PUT /vad-pauses` / `PUT /condition-on-previous-text` / `PUT /overlay-style` / `PUT /gemini-key` / `PUT /youtube-caption-url`
- `PUT /youtube-caption-url` — `{url}`; empty clears. Must be http(s) with a host and no whitespace (422 otherwise). Writes `YOUTUBE_CAPTION_URL` to `.env`, keeping other lines, and publishes `YouTubeCaptionUrlChangedEvent`. The URL is never echoed back; `/mode` shows only `youtube_url_set`.
- `CaptionsStateEvent(is_active: bool)` — default **true** at process start
- `ModeChangedEvent` — pre-service; not verse-by-verse
- `VadSilenceMsEvent` — live; capture updates the running chunker (100–3000 ms)
- `VadPausesEvent` — live; 1 cuts on each pause. 2 waits for a second pause, then one line per phrase
- `ConditionOnPreviousTextEvent` — live; feed the caption just written into the next phrase
- `OverlayStyleEvent` — live over `/ws` (`position`, `font_size_vw` 1–8, `box_width_pct` 10–100, `box_height_pct` 5–100, `inset_vertical_pct` / `inset_horizontal_pct` 0–20, `line_count` 1–6). `line_count` is visible text rows, default 2. A wrapped sentence keeps the newest rows and the oldest row leaves.

Open `/control` in a normal browser, **not** as an OBS source.

## Event bus

In-process asyncio pub/sub. Dispatch by exact type. `publish` isolates handler exceptions. No singleton; `main.py` injects one bus.

Producer-owned events:

- `AudioChunkEvent` — `capture_audio`
- `SubtitleEvent` — `translate_speech` (read by `broadcast_subtitles` and `youtube_captions`)
- `CaptionsStateEvent`, `ModeChangedEvent`, `VadSilenceMsEvent`, `VadPausesEvent`, `ConditionOnPreviousTextEvent`, `OverlayStyleEvent`, `GeminiKeyChangedEvent`, `YouTubeCaptionUrlChangedEvent` — `operator_control`

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

Notable: `AUDIO_DEVICE_NAME` (default `HDMI (Elgato 4K S)`), `AUDIO_DEVICE_INDEX`, VAD (`VAD_MIN_SILENCE_MS` default 500), `WHISPER_MODEL` (default `large-v3`), `WHISPER_DEVICE` (default `cuda`), `WHISPER_COMPUTE_TYPE` (default `float16`), `TRANSLATE_MODE` (`ko_to_en` \| `en_to_en` \| `en_to_ko`), queue size, host/port, `YOUTUBE_CAPTION_URL` (optional). Captions active is **not** persisted; always starts on.

## Operator path

1. Church PC: **thewaychurch** sign-in runs `scripts/Metagrafo.cmd` in that user session (not SYSTEM). Captions **ON**, mode `ko_to_en`.
2. OBS: ATEN video + audio + Browser Source `/overlay`. House projector = program.
3. `/control` in a booth browser.
4. Pastor at pulpit → Captions ON. Worship → Captions OFF.
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
