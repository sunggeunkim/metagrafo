# Metagrafo

Local real-time Korean→English speech captions for OBS (v1 also supports English guest transcribe).

Read before implementing:

- `docs/architecture.md` — VSA, slices, captions default OFF, no NLLB in v1
- `docs/aten-obs.md` — UC9020, `ATEN_Stream_to_USB`, OBS encoder
- `docs/hardware-profiles.md` — NVIDIA VRAM heuristics, Mac/CPU

Work is GitHub issues #1–#4 (`ready-for-agent`). Frontier is #1.

## Agent skills

### Issue tracker

Issues live in GitHub Issues on `sunggeunkim/metagrafo`. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical roles mapped 1:1: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
