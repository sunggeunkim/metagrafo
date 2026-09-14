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
4. Install and **run** ATEN Stream to USB Capture. UC9020 and this PC on a private Ethernet LAN (see `docs/aten-obs.md`).
5. Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/). OBS monitors mixer audio into **CABLE Input**; Metagrafo opens **CABLE Output** (`AUDIO_DEVICE_NAME` default `VB-Audio Virtual Cable`).
6. Install OBS Studio.

```powershell
uv run python -m main --list-devices
```

Confirm `CABLE Output (VB-Audio Virtual Cable)` is listed.

```powershell
uv run uvicorn main:create_production_app --factory --host 127.0.0.1 --port 8000
```

First start may download Whisper `large-v3` (~3 GB). To skip that during rehearsal, see **Whisper model** below.

Inject a fake line (no mic):

```powershell
uv run python -c "import httpx; httpx.post('http://127.0.0.1:8000/inject', json={'text': 'Hello, everyone.'})"
```

Tests: `uv run pytest`.

## Rehearsal from a YouTube sermon

Not the Sunday capture path. **ffmpeg must be on PATH.** First run may still download Whisper `large-v3`. A long sermon takes a while (sequential ~12 s chunks).

```powershell
uv run python -m main --youtube "https://www.youtube.com/watch?v=VIDEO_ID"
```

Saves video + wav (kept) under `metagrafo_job/<yyyyMMddHHmm>/` in the current directory, and writes `{VIDEO_ID}.en.txt` there (English lines, one per VAD cut). Optional: `--out captions.txt`, `--file sermon.wav` (wav in only, skip download), `--vad-silence-ms 250`.

## OBS scene

ATEN Stream to USB Capture **is** the RTMP ingest (its **Play** button starts a listener on this PC). Do not install nginx. In Stream to USB: add the UC9020, click **Play**, copy the device **Stream Key**. Leave Play running.

1. **Media Source** (uncheck Local File) → `rtmp://127.0.0.1/live/<that-stream-key>`.
2. **Settings → Audio → Monitoring device** → **CABLE Input**. On **Media**, **Monitor and Output**. Metagrafo captures **CABLE Output**.
3. **Browser Source** → `http://127.0.0.1:8000/overlay`
   - Width **1920**, height **1080**. Then **Transform → Reset Transform** so letters are not stretched.
   - Do not drag the red handles or change source Height to crop captions. Use **Box height (%)** on `/control`.
   - Shutdown source when not visible
   - Transparent page; two completed caption lines in a `/control`-sized box. Not karaoke.
   - After overlay HTML changes, right-click the source → **Refresh**.
4. Keep the Browser Source **above** the video.
5. Fullscreen projector / HDMI to the house = this program.

Booth UI is **not** an OBS source. Open `http://127.0.0.1:8000/control` in a normal browser on the booth PC.

Live (no uvicorn restart): **VAD pause (ms)** — wait after speech before cutting a caption (100–3000, default 2000); **overlay position** (top left / top center / bottom center); **font size (vw)** (1–8, default 2.5); **box width/height (%)** (10–100 / 5–100, default 100 × 30); **edge insets (%)** (0–20, default 1 vertical / 1 horizontal). Whisper model / CUDA / float16 still need a restart.

## Sunday morning (`/control`)

Captions **default ON** when the process starts. Turn **OFF** during worship so Whisper does not run on music.

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

Defaults: `WHISPER_MODEL=large-v3`, `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`, `VAD_MIN_SILENCE_MS=2000`. Env always wins. If compute type is unset, a VRAM heuristic still applies:

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
