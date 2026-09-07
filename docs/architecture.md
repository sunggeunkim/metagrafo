# Architecture

Metagrafo is a single-process, fully local speech pipeline that turns live **program audio** into OBS captions.

OBS on the broadcast PC is the encoder (YouTube and the house projector share that canvas). Audio is the StreamLIVE PGM mix via **ATEN Stream to USB** on Windows (`ATEN_Stream_to_USB`). Silero VAD chunks speech; faster-whisper produces English; a FastAPI WebSocket overlay shows **two completed lines**. Hardware path: `docs/aten-obs.md`. GPU/OS profiles: `docs/hardware-profiles.md`.

## Product locks (v1)

| Topic | Lock |
|---|---|
| Encoder | OBS encodes the destination. UC9020 RTMP alone has **no captions**. |
| Audio | PGM mix. No speech-only bus. |
| Worship | Captions **default OFF**. Volunteer ON at the pulpit, OFF for songs. Mute **stops inference**, not only the OBS source. |
| Presentation | Two-line, utterance-final. **Not karaoke.** ~2–3 s after they stop is OK. |
| Modes | `ko_to_en` (sermon) and `en_to_en` (English guest transcribe). **`en_to_ko` / NLLB deferred.** |
| Code-switch | Stay on `ko_to_en`. `church_vocabulary.txt` → Whisper `initial_prompt`. No verse-by-verse mode flips. |
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
  church_vocabulary.txt
  features/
    capture_audio/          # named input → AudioChunkEvent
    translate_speech/       # AudioChunkEvent → SubtitleEvent (no HTTP)
    broadcast_subtitles/    # /overlay, /ws, /health
    operator_control/       # /control, /mode, CaptionsStateEvent
  docs/
    architecture.md
    aten-obs.md
    hardware-profiles.md
```

Each feature exposes `register(...)`. FastAPI routes live in the slice that owns that surface.

## Feature slices

### `capture_audio`

Opens the **named** recording device (Windows default `ATEN_Stream_to_USB`). Overridable by name/index for a future Mac box.

- Windows: shared WASAPI (not exclusive) so OBS can use the same device.
- Downmix stereo → mono, resample to 16 kHz, 512-sample frames.
- Silero VAD (ONNX): ~200 ms pre-roll, ~800 ms trailing silence, ~12 s cap, drop &lt;~250 ms.
- Subscribe `CaptionsStateEvent`: if inactive, **drop frames, no VAD**.
- PCM on the bus is `bytes` (s16le).
- No FastAPI routes.

### `translate_speech`

Subscribes to `AudioChunkEvent`, `CaptionsStateEvent`, and mode. No `/control`.

| Mode | Whisper language | Whisper task | On screen |
|---|---|---|---|
| `ko_to_en` | `ko` | `translate` | English |
| `en_to_en` | `en` | `transcribe` | English (unchanged) |

- `large-v3` (`turbo` ignores `task="translate"`). Inference in `asyncio.to_thread`.
- `church_vocabulary.txt` read at **startup** → `initial_prompt`.
- Device profile: see `docs/hardware-profiles.md`. Env always wins.
- Queue max 4, **drop oldest**. Empty output is not published.
- Captions off: **empty the queue**. When Whisper returns, if inactive, **do not publish**.

### `broadcast_subtitles`

- `WS /ws` — caption and mode JSON
- `GET /overlay` — OBS Browser Source (no booth buttons)
- `GET /health` — liveness, client count, captions active, mode, model/device/VRAM

Two-line overlay: upper = previous (dim), lower = current. New `text` shifts current up. Captions OFF **clears both lines**. Fade after a quiet interval.

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

### `operator_control`

Owns the booth UI. Publishes only.

- `GET /control` — Captions ON/OFF, Mode ko_to_en, Mode en_to_en (Guest)
- `GET /mode` / `PUT /mode`
- `CaptionsStateEvent(is_active: bool)` — default **false** at process start
- `ModeChangedEvent` — pre-service; not verse-by-verse

Open `/control` in a normal browser, **not** as an OBS source.

## Event bus

In-process asyncio pub/sub. Dispatch by exact type. `publish` isolates handler exceptions. No singleton; `main.py` injects one bus.

Producer-owned events:

- `AudioChunkEvent` — `capture_audio`
- `SubtitleEvent` — `translate_speech`
- `CaptionsStateEvent`, `ModeChangedEvent` — `operator_control`

## Process model

One process, one asyncio loop, uvicorn.

| Work | Where |
|---|---|
| PyAudio read | capture thread |
| Silero VAD | capture thread (only if captions on) |
| faster-whisper | `asyncio.to_thread` |
| WebSocket fan-out | asyncio |

Startup: operator_control + translate + broadcast, then capture. Shutdown: stop capture first.

## Settings

Env (`core/settings.py`). Profile fills unset whisper fields only.

Notable: `AUDIO_DEVICE_NAME` (default `ATEN_Stream_to_USB`), `AUDIO_DEVICE_INDEX`, VAD, `WHISPER_MODEL` / `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE`, `TRANSLATE_MODE` (`ko_to_en` \| `en_to_en`), queue size, host/port. Captions active is **not** persisted; always starts off.

## Operator path

1. Start uvicorn. Captions **OFF**, mode `ko_to_en`.
2. OBS: ATEN video + audio + Browser Source `/overlay`. House projector = program.
3. `/control` in a booth browser.
4. Pastor at pulpit → Captions ON. Worship → Captions OFF.
5. English guest: set **en_to_en** before service.
6. Saturday: edit `church_vocabulary.txt`; restart Sunday morning.
7. Rehearsal hitch: `WHISPER_MODEL=medium`, restart. Never hot-swap on the GPU.

## Out of scope (v1)

- `en_to_ko` / NLLB
- Karaoke / partials / word highlight
- Application database
- whisper.cpp Metal
- Live VRAM/model swap
- Speech-only aux mix
- Cloud ASR
