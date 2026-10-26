# Metagrafo

Local Korean→English speech captions for OBS. An operator should be able to run a service from this file alone.

Audio is already arriving on this PC as a recording device. Metagrafo translates that speech and serves a transparent overlay page. OBS places that page on the video.

Package management is **[uv](https://docs.astral.sh/uv/)**. Architecture (for developers): `docs/architecture.md`. GPU/OS defaults: `docs/hardware-profiles.md`.

## Where the captions appear

The overlay is `http://127.0.0.1:8000/overlay`. Captions show up wherever OBS sends the program that contains that page: a stream, a projector, or another display of the same canvas. They can also go to YouTube as closed captions; see [YouTube closed captions](#youtube-closed-captions-optional).

## Another Windows computer

Download the repo zip from GitHub and unzip it. Double-click `Start Metagrafo.cmd`.

The first time, that file downloads Python and the Gemini pieces. It needs the internet. Leave the window open. When it finishes, the booth page opens in the browser. Paste the Gemini key and click **Save key**. To get a key, sign in at [Google AI Studio](https://aistudio.google.com/api-keys) with a Google account and create an API key. Then pick the PC's sound input under **Audio input** and click **Use this input**. Both are saved, so the next start remembers them.

The next time, double-click the same file. Close the window after the page opens. Metagrafo keeps running.

After you unzip a newer copy over the same folder, the same file installs any new pieces before it starts. A start that stopped halfway, for example when the internet dropped, picks up again on the next double-click. While it runs, **Show logs** on the booth page shows `metagrafo.log` and follows new lines, and **Show overlay** previews what OBS shows. To stop the captions, click **Stop captions**; **Start captions** brings them back. If Metagrafo does not start, the window says so and opens `metagrafo.log` in Notepad. The reason is at the end of that file.

## Windows install

For this computer, where uv is already the way Metagrafo is developed:

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone this repo and `cd` into it.
3. `uv sync`
4. Install OBS Studio.

```powershell
uv run python -m main --list-devices
uv run python -m main
```

The default recording device is **`HDMI (Elgato 4K S)`**. To use another input, see **Input device**.

Do not use `uv run uvicorn …`. Windows Smart App Control blocks `.venv\Scripts\uvicorn.exe` (`os error 4551`). `python -m main` loads uvicorn inside `python.exe`, which is allowed.

First start may download Whisper `large-v3` (~3 GB). To skip that during rehearsal, see **Whisper model** below.

Inject a fake line (no mic):

```powershell
uv run python -c "import httpx; httpx.post('http://127.0.0.1:8000/inject', json={'text': 'Hello, everyone.'})"
```

Tests: `uv run pytest`.

## Input device

Metagrafo opens one recording device. The easiest way to change it is **Audio input** on `/control`: pick a name, then **Use this input**. Capture switches in about a second, and the choice is saved as `AUDIO_DEVICE_NAME` in `.env`, so the next start (including sign-in) uses it too. **Refresh list** picks up a device plugged in after the page opened. Under the list, the page says which input it is listening to, or why the one you picked would not open. If it would not open, pick another.

Without the booth page, list the names this PC exposes:

```powershell
uv run python -m main --list-devices
```

Set `AUDIO_DEVICE_NAME` to a unique substring of one of those names. Matching is case-insensitive. Put it in `.env` so each start uses it, including sign-in. See **Settings**.

| Input | `AUDIO_DEVICE_NAME` |
|---|---|
| Elgato 4K S (HDMI) | `HDMI (Elgato 4K S)` — also the value used when the variable is unset |
| ATEN Stream to USB | `ATEN_Stream_to_USB` |

The name has to appear in `--list-devices`. A name that is not listed stops capture and prints the names that are available.

A shell variable applies to that window and overrides `.env`:

```powershell
$env:AUDIO_DEVICE_NAME = "ATEN_Stream_to_USB"
uv run python -m main
```

Restart Metagrafo after changing `.env` or the shell variable by hand. A running process keeps the device it opened until you pick another on `/control`. A shell variable still wins over the `.env` line that `/control` writes, at the next start. The sign-in launcher does not set this variable.

When the same name is listed more than once, a name containing `ATEN_Stream_to_USB` opens that device. Other names prefer the 2-channel WASAPI copy. `AUDIO_DEVICE_INDEX` selects one PortAudio index and overrides the name. Indexes move when devices are plugged or unplugged, so leave it unset and use the name. While it is set, `/control` refuses to pick an input.

On macOS, set `AUDIO_DEVICE_NAME` the same way. Typical names are BlackHole, an aggregate device, or a USB interface.

## Settings

Metagrafo reads a file named `.env` in the repo root (next to `main.py`) at startup. Create that file if it is missing. `.gitignore` already ignores it.

Write one setting per line, as `NAME=value`. Leave out `export`, and leave out spaces around `=`. A value may contain spaces. Lines look like this:

```
AUDIO_DEVICE_NAME=HDMI (Elgato 4K S)
CAPTION_ENGINE=gemini_live
GEMINI_API_KEY=your-key-here
```

Save the file, then start Metagrafo again. A process that is already running keeps the settings it loaded. A variable set in the shell overrides the same name in `.env` for that window:

```powershell
$env:AUDIO_DEVICE_NAME = "ATEN_Stream_to_USB"
uv run python -m main
```

## Caption engine

With `CAPTION_ENGINE` unset, Metagrafo uses local Whisper. See **Whisper model**.

Google Gemini Live translates the live audio when `.env` has both of these lines:

```
CAPTION_ENGINE=gemini_live
GEMINI_API_KEY=your-key-here
```

The model is `gemini-3.5-live-translate-preview`. Korean speech is translated to English while it is still coming in. With Gemini on, booth control also offers **en_to_ko**, which captions an English speaker in Korean. Get the key from [Google AI Studio](https://aistudio.google.com/api-keys). The key belongs only in `.env`. An empty key, or any other `CAPTION_ENGINE` value, stays on Whisper.

## Sign-in autostart

Optional. Copy `scripts/Metagrafo.cmd` into the Startup folder so Metagrafo starts when you sign in:

`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Metagrafo.cmd`

Run it in the signed-in desktop session. The recording device is visible only there. A task that runs as SYSTEM before sign-in cannot open it.

In that Startup copy, set the `cd /d` line to this repo. The launcher titles the console **Metagrafo**, waits about 15 seconds so desktop audio is up, then runs `uv run python -m main`. If the process stops, the window prints the exit code and stays open (`pause`).

`uv` must be on the user PATH. Leave the command as `python -m main`. Smart App Control still blocks `uvicorn.exe`.

Every sign-in uses the caption engine and input device in `.env`. Captions still default **ON**, including weekdays. Turn them **OFF** on `/control` when you are not captioning a sermon.

Hold **Shift** during sign-in to skip Startup items for that logon. Delete the Startup `.cmd` to turn auto-start off. An edit to `scripts/Metagrafo.cmd` does nothing until that file is copied back over the Startup copy.

## Rehearsal from a YouTube sermon

Not the Sunday capture path. **ffmpeg must be on PATH.** First run may still download Whisper `large-v3`. A long sermon takes a while (sequential ~12 s chunks).

```powershell
uv run python -m main --youtube "https://www.youtube.com/watch?v=VIDEO_ID"
```

Saves video + wav (kept) under `metagrafo_job/<yyyyMMddHHmm>/` in the current directory, and writes `{VIDEO_ID}.en.txt` there (English lines, one per VAD cut). Optional: `--out captions.txt`, `--file sermon.wav` (wav in only, skip download), `--vad-silence-ms 250`, `--pauses 2`, `--condition-on-previous-text`, `--start 1`, `--end 3`. `--start` and `--end` are minutes into the wav. Omit them to translate the whole file (`--start` defaults to 0, `--end` to the end). `--pauses` is 1 or 2 (default 1). Previous text is on unless you pass `--no-condition-on-previous-text`. `--condition-on-previous-text` forces it on. A YouTube URL still downloads the full video; the flags only limit translation.

## OBS scene

Put the video in OBS the way you already do. Add the captions as a Browser Source on top of it.

1. **Browser Source** → `http://127.0.0.1:8000/overlay`
   - Width **1920**, height **1080**. Then **Transform → Reset Transform** so letters are not stretched.
   - Do not drag the red handles or change source Height to crop captions. Use **Box height (%)** on `/control`.
   - Shutdown source when not visible
   - Transparent page. **Lines on screen** (default **2**, range 1–6) is how many text rows stay visible. A longer sentence wraps, and the oldest row leaves when the next one starts. Not karaoke.
   - After overlay HTML changes, right-click the source → **Refresh**.
2. Keep the Browser Source **above** the video.
3. A stream, projector, or other display of this program shows the same captions.

Booth UI is **not** an OBS source. Open `http://127.0.0.1:8000/control` in a normal browser on the booth PC.

Live (no uvicorn restart): **VAD pause (ms)** — wait after speech before a pause counts (100–3000, default 500); **pauses per caption** (1 or 2, default 1; 2 waits for a second pause, then one line per phrase); **previous text** (on by default; gives Whisper the caption it just wrote; turn off on `/control`); **lines on screen** (1–6, default 2; a longer sentence scrolls the oldest row off); **overlay position** (top left / top center / bottom center); **font size (vw)** (1–8, default 2.5); **box width/height (%)** (10–100 / 5–100, default 100 × 30); **edge insets (%)** (0–20, default 1 vertical / 1 horizontal). Whisper model / CUDA / float16 still need a restart. A file run can set pauses with `--pauses 2`. Previous text is already on; turn it off with `--no-condition-on-previous-text`.

## Sunday morning (`/control`)

Captions **default ON** when the process starts. Turn **OFF** during worship so Whisper does not run on music.

| When | Action |
|---|---|
| Pre-service / worship / hymns | Leave captions stopped (**Stop captions**). Do not caption PGM music — Whisper will hallucinate. |
| Pastor at the pulpit | **Start captions**. Mode **ko_to_en** for a Korean sermon. |
| Songs again | **Stop captions**. Stops **new** capture and queued Whisper jobs. Not the same as hiding the OBS source. |
| English guest speaker | Before service, set mode **en_to_en (Guest)**. English stays English. For Korean captions, set **en_to_ko (Guest → Korean)** instead; it needs a saved Gemini key. |

Do **not** flip language verse-by-verse. Stay on `ko_to_en` for a Korean sermon.

Hiding the OBS Browser Source does **not** rest the GPU. Use **Stop captions** on `/control`. That drains the queue and drops frames so worship is not transcribed. An utterance **already on the GPU** can still finish (Whisper cannot be cancelled mid-chunk); that result is **discarded** and will not appear on the overlay. GPU load drops after that current chunk, not instantly.

## Whisper model (NVIDIA)

Defaults: `WHISPER_MODEL=large-v3`, `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`, `VAD_MIN_SILENCE_MS=500`. Env always wins. If compute type is unset, a VRAM heuristic still applies:

| VRAM | Default model | Default compute |
|---|---|---|
| &lt; 6 GB | `medium` | `int8` |
| 6–8 GB | `large-v3` | `int8` |
| 8–12 GB | `large-v3` | `int8` (or `float16` if OBS NVENC still has headroom) |
| ≥ 12 GB | `large-v3` | `float16` |

On macOS / no CUDA: device `cpu`, compute `int8`.

Rehearsal protocol: if OBS NVENC hitchs or the laptop throttles, set `WHISPER_MODEL=medium` and **restart** the Python process. Do not hot-swap the model during a live stream.

```powershell
$env:WHISPER_MODEL = "medium"
# optional:
# $env:WHISPER_DEVICE = "cuda"   # or cpu
# $env:WHISPER_COMPUTE_TYPE = "int8"
uv run python -m main
```

`GET /health` reports `model`, `device`, `compute_type`, `vram_gb`, captions on/off, and mode.

To run the model on the Hermeneia GPU box instead of this PC, set `HERMENEIA_URL` (for example `https://hermeneia.stugen.net`) and `HERMENEIA_TOKEN`. `HERMENEIA_MODEL` defaults to `whisper` and stays fixed for the process. Leave the URL unset to keep local Whisper.

## YouTube closed captions (optional)

Metagrafo can also send each finished caption line to YouTube as **closed captions** (the CC button). Viewers can turn them on or off, and they stay on the recording as a caption track. The OBS overlay keeps working either way.

1. YouTube Studio → your stream → **Stream settings**. Set latency to **Normal** (YouTube only takes captions in Normal latency).
2. Turn on **Closed captions**, choose **POST captions to URL**, and set **Source language** to the caption language: **English**, or **Korean** if you caption with **en_to_ko**.
3. Copy the **Captions Ingestion URL** (`http://upload.youtube.com/closedcaption?cid=…`). Treat it like the stream key.
4. On booth control (`/control`), paste it under **YouTube captions** and click **Save URL**. Sending starts at once, no restart. The URL is saved to `.env` for the next time, and the page shows how many lines were sent. **Stop sending** clears it.

Or put it in `.env` yourself (see **Settings**), then restart Metagrafo:

```
YOUTUBE_CAPTION_URL=http://upload.youtube.com/closedcaption?cid=...
```

The URL follows the stream key, so a reused stream key keeps the same URL. Leave it unset to send nothing.

Only finished lines are sent. The Gemini live bottom line is sent once it is committed, not while it grows. Stopped captions send nothing, because no lines are made. Each line is stamped with when it was spoken: for Whisper, when its audio chunk began; for Gemini live, when the line's first English words arrived (a second or two after the Korean). Keep this PC's clock synced (Windows **Set time automatically**).

YouTube refuses lines until the stream is live. Metagrafo logs one warning when that starts and one line when YouTube accepts captions again. `GET /youtube-captions` shows `sent`, `failed`, and the last HTTP status.

## Mac

Same app. Choose the recording device with **Input device**. Whisper runs on **CPU** until a Metal backend exists. Overlay URL and `/control` are unchanged.
