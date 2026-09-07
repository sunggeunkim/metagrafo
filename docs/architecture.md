# Architecture

Metagrafo is a single-process, fully local speech pipeline that turns live microphone audio into OBS captions.

It captures an ATEN USB microphone, chunks speech with Silero VAD, runs faster-whisper, optionally translates, and broadcasts subtitle JSON over a FastAPI WebSocket that an OBS Browser Source consumes.

## Output-language knob

The operator picks what appears on screen. The mode can change at runtime (`PUT /mode` or the `/control` page). No process restart.

| Mode | Spoken | On screen | How |
|---|---|---|---|
| `ko_to_en` (default) | Korean | English | faster-whisper `task="translate"` |
| `en_to_ko` | English | Korean | faster-whisper `task="transcribe"`, then local NLLB |
| `transcribe` | whatever is spoken | same language | faster-whisper `task="transcribe"` only |

Whisper `task="translate"` only produces English. English→Korean therefore needs a second local model (NLLB-200 distilled 600M via CTranslate2). NLLB is lazy-loaded on first switch to `en_to_ko`.

## Vertical Slice Architecture

The codebase is organized by **feature**, not by layer.

Each feature owns its events, I/O, and runtime loop. Features talk only through a typed asyncio event bus. Downstream features may import **events** from upstream features. They must not import each other's internals.

**Do not create** shared layers named `controllers/`, `services/`, `repositories/`, `models/`, `domain/`, `infrastructure/`, `api/`, or `schemas/`. No generic `*Service`, `*Controller`, or `*Repository` classes.

Allowed shared code:

- `core/event_bus.py` — typed asyncio pub/sub
- `core/settings.py` — env-backed bootstrap config (not a service layer)
- `main.py` — composition root (constructs the bus, registers slices, starts FastAPI)

```
ATEN mic
    │
    ▼
capture_audio  ──AudioChunkEvent──►  translate_speech  ──SubtitleEvent──►  broadcast_subtitles  ──►  OBS
 (PyAudio + Silero VAD)              (whisper ± NLLB,                  (FastAPI /ws + overlay)
                                      mode knob)
```

Language policy lives only in `translate_speech`. Capture does not know about modes. Broadcast renders `SubtitleEvent.text` and does not decide how that text was produced.

## Layout

```
metagrafo/
  main.py                          # composition root
  core/
    event_bus.py
    settings.py
  features/
    capture_audio/                 # mic → AudioChunkEvent
    translate_speech/              # AudioChunkEvent → SubtitleEvent; owns /mode and /control
    broadcast_subtitles/           # SubtitleEvent → WebSocket JSON + overlay
```

Each feature exposes `register(app, bus, settings)` and is wired from `main.py`. FastAPI routes belong to the slice that owns that HTTP surface, not a global router package.

## Feature slices

### `capture_audio`

Opens the ATEN microphone with PyAudio (16 kHz, mono, 512-sample frames), runs Silero VAD (ONNX), and publishes one `AudioChunkEvent` per utterance.

- Device match: name contains `ATEN`, overridable by index or name in settings.
- Capture is blocking, so it runs on a dedicated thread and hops into asyncio with `loop.call_soon_threadsafe`.
- VAD: ~200 ms pre-roll, ~800 ms trailing silence, ~12 s hard cap, drop utterances shorter than ~250 ms.
- PCM on the bus is `bytes` (s16le), not a numpy array, so the event stays frozen and cheap to copy.
- No FastAPI routes.

### `translate_speech`

Subscribes to `AudioChunkEvent`, runs ASR (± translation) according to the **current** mode, publishes `SubtitleEvent`. Owns the output-language knob.

| Mode | Whisper language | Whisper task | NLLB | Target language |
|---|---|---|---|---|
| `ko_to_en` | `ko` | `translate` | no | `en` |
| `en_to_ko` | `en` | `transcribe` | yes, `eng_Latn` → `kor_Hang` | `ko` |
| `transcribe` | settings (`en` / `ko` / auto) | `transcribe` | no | spoken language |

Whisper is `large-v3` with INT8 (`turbo` ignores `task="translate"`). Inference runs in `asyncio.to_thread`. A bounded queue (default 4) **drops oldest** when the translator falls behind so live captions stay current.

Knob HTTP (operator machine, not the OBS overlay):

- `GET /mode` / `PUT /mode` — read or change mode; `PUT` publishes `ModeChangedEvent`
- `GET /control` — three-button page for the operator

A mode change does not cancel the in-flight chunk. It applies when the next `AudioChunkEvent` is dequeued.

### `broadcast_subtitles`

Fans `SubtitleEvent` and `ModeChangedEvent` to every connected WebSocket client.

- `WS /ws` — JSON captions for OBS
- `GET /overlay` — transparent Browser Source page (viewer-facing; no mode buttons)
- `GET /health` — liveness plus client count and current mode

OBS: add a Browser source pointing at `http://127.0.0.1:8000/overlay` (1920×1080).

Subtitle payload:

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

## Event bus

`core/event_bus.py` is in-process asyncio pub/sub. No Redis, no extra broker.

- Dispatch by exact event type (`type(event)`), not inheritance.
- `publish` copies the handler list, then `asyncio.gather(..., return_exceptions=True)`.
- A failing handler is logged; it does not cancel siblings or the publisher.
- `subscribe` returns an unsubscribe callable.
- No global singleton. `main.py` constructs one `EventBus` and injects it.

Producer-owned events:

- `AudioChunkEvent` — `capture_audio`
- `SubtitleEvent`, `ModeChangedEvent` — `translate_speech`

## Process model

One OS process, one asyncio loop, FastAPI via uvicorn.

| Work | Where | Why |
|---|---|---|
| PyAudio `stream.read` | dedicated thread | blocking I/O |
| Silero VAD | capture thread | cheap, stays next to the frames |
| faster-whisper / NLLB | `asyncio.to_thread` | CTranslate2 would stall the event loop |
| WebSocket fan-out | asyncio | native to FastAPI |

Startup order: translator → broadcaster → capture (subscribers ready before mic events). Shutdown is the reverse.

## Settings

Env-backed (`core/settings.py`). Startup defaults only; the live knob overrides `translate_mode` without a restart.

Notable keys: ATEN device name/index, VAD silence/cap, whisper model/device/compute type, `translate_mode`, `transcribe_language`, NLLB model id, translate queue size, host/port.

## Operator path

1. Start uvicorn. Default mode is `ko_to_en`.
2. OBS Browser Source → `/overlay`.
3. Operator browser → `/control` (not added as an OBS source).
4. Korean speaker → **KO→EN**. English that should stay English → **Transcribe**. English that needs Korean captions → **EN→KO**.
