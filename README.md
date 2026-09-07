# Metagrafo

Local Korean→English speech captions for OBS. Package management is **[uv](https://docs.astral.sh/uv/)**.

## Run (issue #1 overlay + booth control)

```powershell
uv sync
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

- Overlay (OBS Browser Source): http://127.0.0.1:8000/overlay
- Booth control: http://127.0.0.1:8000/control (captions start **OFF**)
- Inject a test line: `uv run python -c "import httpx; httpx.post('http://127.0.0.1:8000/inject', json={'text': 'Hello, everyone.'})"`

Architecture: `docs/architecture.md`. Tests: `uv run pytest`.
