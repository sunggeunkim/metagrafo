# Church setup (2026-10-13)

How The Way Church caption laptop is wired, and how to install and start Metagrafo on it.

The laptop is the Dell XPS that signs in as **thewaychurch**. The repo on that machine is `C:\Users\thewaychurch\metagrafo`. Metagrafo translates the program audio and serves an overlay page. OBS places that page on the Elgato picture and sends the result to the projector.

## How the devices connect

```
Pastor mics and cameras
        |
        v
  ATEN UC9020 (program mix)
        |
        | HDMI  (picture and program audio)
        v
  Elgato USB capture (Game Capture 4K S)
        |
        | USB
        v
  Laptop  (OBS + Metagrafo)
        |
        | HDMI
        v
  Projector
```

| Cable | From | To | What it carries |
|---|---|---|---|
| HDMI | UC9020 program output | Elgato HDMI input | The mixed program: video and audio |
| USB | Elgato | Laptop | That same program, as a camera and a recording device |
| HDMI | Laptop | Projector | The OBS picture with English captions on top |

Leave the projector on the laptop HDMI. The UC9020 program output has no captions. The Elgato HDMI passthrough, if used, is that same clean picture. Captions exist only on the OBS canvas.

On the laptop the Elgato shows up as a camera and as the recording device **`HDMI (Elgato 4K S)`**. Metagrafo listens to that recording device. OBS takes the video from the same card. House speakers stay on the mixer. The laptop does not have to feed the room audio.

## OBS on the laptop

Scene name: **Church (Video + Caption)**. A fullscreen projector of that scene goes to the projector display.

1. **Video Capture Device** → the Elgato. If the source still says **OBS Virtual Camera**, change it to the Elgato.
2. **Browser Source** named **Caption** → `http://127.0.0.1:8000/overlay`
   - Width **1920**, height **1080**, then **Transform → Reset Transform**.
   - Turn the eye on so the source is visible, and keep it above the video.
   - After overlay HTML changes, right-click the source → **Refresh**.
3. Right-click the scene → **Fullscreen Projector (Scene)** → the display that is the projector.

Booth controls are a normal browser page, not an OBS source: `http://127.0.0.1:8000/control`.

## Install uv and the repo

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/). On this laptop it is the per-user WinGet install and is already on the user PATH.
2. Clone the repo and enter it:

```powershell
git clone git@github.com:sunggeunkim/metagrafo.git C:\Users\thewaychurch\metagrafo
cd C:\Users\thewaychurch\metagrafo
uv sync
```

`uv sync` creates `.venv` and installs the app. The first live start can also download Whisper `large-v3` (about 3 GB) if Gemini is not selected.

## `.env`

Create `.env` in the repo root, next to `main.py`, if it is not already there. Git ignores this file. One setting per line. No `export`. No spaces around `=`.

This church runs Google Gemini Live. The file on this laptop already has these two lines. Keep them. Replace the key only when Google issues a new one. Do not commit the file or paste the key into chat or docs.

```
CAPTION_ENGINE=gemini_live
GEMINI_API_KEY=your-google-ai-studio-key
```

Both lines are required. `CAPTION_ENGINE=gemini_live` with an empty key stays on local Whisper. The live model is `gemini-3.5-live-translate-preview`. Korean speech is translated to English while it is still coming in.

The recording device does not need a line in `.env`. With `AUDIO_DEVICE_NAME` unset, Metagrafo opens **`HDMI (Elgato 4K S)`**. To make that choice explicit, add:

```
AUDIO_DEVICE_NAME=HDMI (Elgato 4K S)
```

A name set in PowerShell overrides `.env` for that window only. Sign-in does not set it. Restart Metagrafo after any change. A process that is already running keeps the file it loaded at startup.

Check the name Windows is using:

```powershell
uv run python -m main --list-devices
```

`HDMI (Elgato 4K S)` must be in that list before a service.

## Run it

From the repo root:

```powershell
uv run python -m main
```

Leave this window open. The app serves:

| Page | URL |
|---|---|
| Overlay (OBS Browser Source) | `http://127.0.0.1:8000/overlay` |
| Booth controls | `http://127.0.0.1:8000/control` |
| Health | `http://127.0.0.1:8000/health` |

Use `python -m main`. `uv run uvicorn …` starts `.venv\Scripts\uvicorn.exe`, and Windows Smart App Control on this laptop blocks that program (`os error 4551`).

Stop with Ctrl+C in the console titled Metagrafo, or close that window.

### Sign-in

This laptop also starts Metagrafo when **thewaychurch** signs in. Windows runs the copy in the Startup folder:

`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Metagrafo.cmd`

That copy matches `scripts/Metagrafo.cmd`. It waits about 15 seconds, then runs `uv run python -m main` from `C:\Users\thewaychurch\metagrafo`. An edit to the repo file does nothing until it is copied over the Startup file.

Hold Shift during sign-in to skip Startup items once. Captions come up **ON**. Turn them **OFF** on `/control` during music, and **ON** when the sermon starts. Hiding the OBS Caption source does not stop translation.

## Sunday

| When | Action |
|---|---|
| Before the sermon, and during music | **Captions OFF** on `/control` |
| Korean sermon | **Captions ON**. Mode **ko_to_en** |
| English guest | Mode **en_to_en (Guest)** before they speak |

Stay on `ko_to_en` for a Korean sermon. The projector shows captions only while the Caption source is visible and Metagrafo is running.
