# Metagrafo

Local Korean→English speech captions for OBS. An operator should be able to run a service from this file alone.

Package management is **[uv](https://docs.astral.sh/uv/)**. Architecture (for developers): `docs/architecture.md`. Hardware routing: `docs/aten-obs.md`. GPU/OS defaults: `docs/hardware-profiles.md`.

## Captions only leave this PC if OBS encodes

OBS on this computer is the YouTube encoder. The ATEN UC9020 (StreamLIVE HD) mix reaches Windows through **ATEN Stream to USB**. Metagrafo captions sit on the OBS canvas.

If the UC9020 RTMPs to YouTube by itself, the audience gets **no captions**. The house HDMI projector is the **same OBS program**, so English captions appear in the sanctuary too.

## Windows install (broadcast laptop)

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone this repo and `cd` into it.
3. `uv sync`
4. Install and **run** ATEN Stream to USB Capture. UC9020 and this PC must be on the same LAN. The virtual recording device is typically named `ATEN_Stream_to_USB`.
5. Install OBS Studio.

```powershell
uv run python -m main --list-devices
```

Confirm `ATEN_Stream_to_USB` is listed. If it is missing, Stream to USB is not running or the mixer is off the LAN.

```powershell
uv run uvicorn main:create_production_app --factory --host 127.0.0.1 --port 8000
```

First start may download Whisper `large-v3` (~3 GB). To skip that during rehearsal, see **Whisper model** below.

Inject a fake line (no mic):

```powershell
uv run python -c "import httpx; httpx.post('http://127.0.0.1:8000/inject', json={'text': 'Hello, everyone.'})"
```

Tests: `uv run pytest`.

## OBS scene

1. **Video Capture Device** → Stream to USB webcam (`ATEN_Stream_to_USB` or whatever Windows shows).
2. **Audio Input Capture** → the same `ATEN_Stream_to_USB` recording device (or “use custom audio device” on the video source). Shared WASAPI so Metagrafo can open it too.
3. **Browser Source** → `http://127.0.0.1:8000/overlay`
   - Width **1920**, height **1080**
   - Shutdown source when not visible
   - Transparent page; two completed caption lines (previous dim, current bright). Not karaoke.
4. Keep the Browser Source **above** the video.
5. Fullscreen projector / HDMI to the house = this program.

Booth UI is **not** an OBS source. Open `http://127.0.0.1:8000/control` in a normal browser on the booth PC.

## Sunday morning (`/control`)

Captions **default OFF** when the process starts. That is fail-safe for opening music.

| When | Action |
|---|---|
| Pre-service / worship / hymns | Leave **Captions OFF**. Do not caption PGM music — Whisper will hallucinate. |
| Pastor at the pulpit | **Captions ON**. Mode **ko_to_en** for a Korean sermon. |
| Songs again | **Captions OFF**. Stops **new** capture and queued Whisper jobs. Not the same as hiding the OBS source. |
| English guest speaker | Before service, set mode **en_to_en (Guest)**. English stays English. No Korean captions in v1. |

Do **not** flip language verse-by-verse. `ko_to_en` plus `church_vocabulary.txt` handles English names and book titles inside a Korean sermon.

Hiding the OBS Browser Source does **not** rest the GPU. Use **Captions OFF** on `/control`. That drains the queue and drops frames so worship is not transcribed. An utterance **already on the GPU** can still finish (Whisper cannot be cancelled mid-chunk); that result is **discarded** and will not appear on the overlay. GPU load drops after that current chunk, not instantly.

## Saturday: vocabulary file

Edit `church_vocabulary.txt` in Notepad (pastor name, series title, extra book names). The file is read at **startup** as Whisper’s `initial_prompt`. Restart Metagrafo Sunday morning for changes to apply.

## Whisper model (NVIDIA)

Env always wins. If unset, Metagrafo picks from OS + NVIDIA VRAM:

| VRAM | Default model | Default compute |
|---|---|---|
| &lt; 6 GB | `medium` | `int8` |
| 6–8 GB (this XPS 3060) | `large-v3` | `int8` |
| 8–12 GB | `large-v3` | `int8` (or `float16` if OBS NVENC still has headroom) |
| ≥ 12 GB | `large-v3` | `float16` |

On macOS / no CUDA: device `cpu`, compute `int8`.

Rehearsal protocol: if OBS NVENC hitchs or the laptop throttles, set `WHISPER_MODEL=medium` and **restart** the Python process. Do not hot-swap the model during a live stream.

```powershell
$env:WHISPER_MODEL = "medium"
# optional:
# $env:WHISPER_DEVICE = "cuda"   # or cpu
# $env:WHISPER_COMPUTE_TYPE = "int8"
uv run uvicorn main:create_production_app --factory --host 127.0.0.1 --port 8000
```

`GET /health` reports `model`, `device`, `compute_type`, `vram_gb`, captions on/off, and mode.

## Mac

Same app. ATEN Stream to USB is **Windows-only**. Set `AUDIO_DEVICE_NAME` to BlackHole, an aggregate device, or a USB interface. Whisper runs on **CPU** until a Metal backend exists. Overlay URL and `/control` are unchanged.
