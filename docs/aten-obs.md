# ATEN StreamLIVE + OBS integration

How Metagrafo sits next to an **ATEN UC9020 StreamLIVE HD** (often called US9020) and **OBS Studio**.

The UC9020 is a hardware AV mixer. It is **not** a USB microphone plugged into the PC. Audio and video reach Windows through **ATEN Stream to USB**, which exposes a virtual webcam (UVC) and virtual microphone (UAC). That virtual audio device is typically named **`ATEN_Stream_to_USB`**. That is the device Metagrafo should capture.

## Signal flow

```
Pastor mic / HDMI cameras
        │
        ▼
   ATEN UC9020 (mix + optional RTMP)
        │  LAN
        ▼
 ATEN Stream to USB  ── virtual video ──►  OBS Video Capture Device
                     ── virtual audio ──►  OBS Audio Input Capture
                     ── same audio ────►  Metagrafo (PyAudio) → two-line captions
                                              │
                                              ▼
                                    OBS Browser Source /overlay
                                              │
                                              ▼
                                         YouTube / record
```

Captions appear on the public stream **only if OBS is encoding that destination**. If the UC9020 RTMPs to YouTube by itself, the overlay stays on this PC.

## Prerequisites

1. UC9020 and this PC on the **same LAN**.
2. **ATEN Stream to USB Capture** installed and **running** (virtual devices do not exist otherwise).
3. OBS Studio on this PC.
4. Metagrafo listening on `http://127.0.0.1:8000`.

## What Metagrafo opens

Default input name: **`ATEN_Stream_to_USB`**.

Settings still allow `AUDIO_DEVICE_NAME` / `AUDIO_DEVICE_INDEX` override. Do not grab the first device whose name merely contains `ATEN` if more than one ATEN endpoint exists; prefer `ATEN_Stream_to_USB`.

The virtual device is usually **48 kHz stereo**. Silero VAD and Whisper want **16 kHz mono**. Capture must:

- open in **shared WASAPI** (not exclusive), so OBS can use the same device
- **downmix stereo → mono**
- **resample to 16 kHz**

If Windows still exclusive-locks the device, copy it with a virtual cable and point Metagrafo at the copy.

`--list-devices` should print the Windows recording name so an operator can confirm `ATEN_Stream_to_USB` is present.

## OBS scene

1. **Video Capture Device** → `ATEN_Stream_to_USB` (or the Stream to USB webcam name Windows shows).
2. **Audio Input Capture** → the matching `ATEN_Stream_to_USB` recording device (or “use custom audio device” on the video source).
3. **Browser Source** → `http://127.0.0.1:8000/overlay`, 1920×1080, shutdown when not visible. Transparent background; two-line completed captions, not karaoke.
4. Keep this Browser Source **above** the video in the scene.

Operator control stays at `http://127.0.0.1:8000/control` in a normal browser, **not** as an OBS source.

## What this does not change

Feature slices, event bus, two-line overlay, and Whisper/NLLB are unchanged. Only the capture **source** is a virtual UAC from StreamLIVE’s mix (pastor mic + HDMI), not a dedicated USB mic.

## Failure modes

| Symptom | Likely cause |
|---|---|
| No ATEN device in `--list-devices` | Stream to USB not running, or UC9020 off the LAN |
| Device found, silence / no VAD | Wrong ATEN endpoint; OBS holding exclusive mode; muted in Windows |
| Garbled / chipmunk audio | Sample rate not resampled (48 kHz treated as 16 kHz) |
| Overlay locally, missing on YouTube | Stream is leaving from the UC9020 RTMP path, not OBS |
| GPU hitch while captioning | OBS NVENC + Whisper on the 6 GB laptop 3060 (65 W); not an ATEN issue |
