# Hardware profiles

v1 **ships on Windows + NVIDIA**. The app must not assume a 6 GB / 65 W RTX 3060, and a later Mac box must not require a rewrite.

Env always wins: `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`. The profile runs at startup only when those are **unset**. It never changes mid-service. `/health` logs the chosen model, device, compute type, and VRAM.

## NVIDIA VRAM (Windows, CUDA)

| Detected VRAM | Default model | Default compute |
|---|---|---|
| &lt; 6 GB | `medium` | `int8` |
| 6–8 GB (Dell XPS 3060 laptop) | `large-v3` | `int8` (fallback `int8_float16` if int8 will not load) |
| 8–12 GB | `large-v3` | `int8`, or `float16` if there is headroom after OBS NVENC |
| ≥ 12 GB | `large-v3` | `float16` |

Rehearsal: if NVENC hitchs or the laptop throttles, set `WHISPER_MODEL=medium` and **restart**. Do not unload/reload CUDA weights during a live stream.

`turbo` is not a KO→EN option (`task="translate"` is ignored).

## macOS

- **No CUDA.** `whisper_device=auto` → `cpu`.
- v1 ASR is still **faster-whisper** (CTranslate2) on CPU. No whisper.cpp / Metal in v1. A later Metal backend can replace the body of the whisper loader without touching capture or overlay.
- Apple Silicon CPU can run `medium` or `large-v3` int8 for utterance-final captions; pick with the same env keys.
- Capture uses PortAudio/Core Audio. Shared WASAPI is Windows-only; the capture slice picks host API per OS. Stereo downmix and 16 kHz resample stay.
- Silero VAD ONNX and the OBS Browser Source overlay are unchanged.

## Audio device

Capture always means **open the recording device named in settings**, never a hardcoded ATEN driver.

| OS | Default `AUDIO_DEVICE_NAME` | Notes |
|---|---|---|
| Windows | `ATEN_Stream_to_USB` | Requires ATEN Stream to USB Capture running. See `docs/aten-obs.md`. |
| macOS | (none — set explicitly) | Stream to USB is **Windows-only**. Use BlackHole, an aggregate device, or a USB interface. UC9020 → Mac is a routing problem, not a second capture stack. |

## Same-machine OBS

The broadcast PC also runs OBS NVENC (and may fullscreen-project to the house). Whisper must leave VRAM for encoding. Captions OFF must **stop inference** so worship does not fight the 3060.
