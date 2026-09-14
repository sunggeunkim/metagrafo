# ATEN StreamLIVE + OBS integration

How Metagrafo sits next to an **ATEN UC9020 StreamLIVE HD** (often called US9020) and **OBS Studio**.

The UC9020 is a hardware AV mixer. It is **not** a USB microphone. On **Windows**, **ATEN Stream to USB** pulls the mixer over LAN. Video (and embedded HDMI audio) usually arrive in OBS as an RTMP **Media Source**. Stream to USB’s virtual mic (`ATEN_Stream_to_USB`) often never appears; Metagrafo’s default capture is **`VB-Audio Virtual Cable`** (`CABLE Output`) after OBS monitors **Media** into **CABLE Input**.

That software is **Windows-only**. On a Mac, set `AUDIO_DEVICE_NAME` to whatever actually carries program audio. See `docs/hardware-profiles.md`.

## Signal flow

```
Pastor mic / HDMI cameras
        │
        ▼
   ATEN UC9020 (mix)
        │  LAN
        ▼
 ATEN Stream to USB  ── RTMP ──►  OBS Media Source (video + HDMI audio)
                     ── OBS monitor ──►  VB-Cable CABLE Input
                     ── CABLE Output ─►  Metagrafo (when captions ON)
                                              │
                                              ▼
                                    OBS Browser Source /overlay
                                              │
                                              ▼
                              YouTube  +  house HDMI projector
                              (same OBS program canvas)
```

Captions reach YouTube and the sanctuary **only if OBS is encoding**. If the UC9020 RTMPs to the CDN by itself, the overlay never leaves this PC. The house projector is the same canvas: English captions appear in the room.

## Prerequisites (Windows)

1. UC9020 and this PC on the **same LAN**.
2. **ATEN Stream to USB Capture** installed and **running**. Add the UC9020, then click **Play**. Play starts Stream to USB’s bundled RTMP listener on this PC (`rtmp://127.0.0.1/live/<stream-key>`). Copy the **Stream Key** from the device in that app. Leave Play running while OBS is open. Metagrafo does not start this listener.
3. OBS Studio on this PC (this is the live encoder).
4. VB-Audio Virtual Cable installed.
5. Metagrafo at `http://127.0.0.1:8000`.

## What Metagrafo opens

Default name: **`VB-Audio Virtual Cable`**. Name match prefers a **2-channel** device so PortAudio does not open the 16-channel MME cable. Override with `AUDIO_DEVICE_NAME` / `AUDIO_DEVICE_INDEX` (for example `Microphone Array` or `ATEN_Stream_to_USB`).

The virtual device is usually **48 kHz stereo**. VAD and Whisper want **16 kHz mono**:

- open **shared WASAPI** (not exclusive) so OBS can share the device
- **downmix stereo → mono**
- **resample to 16 kHz**

If Windows exclusive-locks the device, copy it with a virtual cable.

`--list-devices` must show the Windows recording name.

When captions are **OFF**, Metagrafo **drops frames** and does not run VAD or Whisper. Hiding the OBS Browser Source alone does **not** shed GPU load. Captions start **ON**.

## OBS scene

1. Stream to USB **Play** is on (RTMP ingest is already listening).
2. **Media Source** (uncheck Local File) → `rtmp://127.0.0.1/live/<Stream-to-USB-key>` (mixer video + HDMI audio).
3. OBS **Settings → Audio → Monitoring device** → **CABLE Input**. Media source **Monitor and Output**. Metagrafo opens **CABLE Output**.
4. **Browser Source** → `http://127.0.0.1:8000/overlay`, 1920×1080, shutdown when not visible. Two-line completed captions, not karaoke. Above the video.
5. Fullscreen projector / HDMI to the house = this program (English overlay in the room).

Booth UI: `http://127.0.0.1:8000/control` in a **normal browser**, not an OBS source. Captions start ON. OFF for worship; ON at the pulpit.

## Failure modes

| Symptom | Likely cause |
|---|---|
| No `CABLE Output` in `--list-devices` | VB-Cable not installed |
| No ATEN device in `--list-devices` | Normal on this booth; capture uses VB-Cable, not UAC |
| OBS Media Source black / no audio | Stream to USB **Play** not running, wrong stream key, or mixer not pushing to this PC |
| Device found, silence | Wrong endpoint; exclusive WASAPI; muted in Windows; captions still OFF |
| Garbled / chipmunk audio | 48 kHz treated as 16 kHz |
| Overlay locally, missing on YouTube | Stream leaving from UC9020 RTMP, not OBS |
| Garbage English during songs | Captions left ON on PGM worship; turn OFF |
| GPU hitch while captioning | OBS NVENC + Whisper on a small NVIDIA card; try `WHISPER_MODEL=medium` and restart |
