# Changelog

All notable changes to this project will be documented in this file.
This project uses Semantic Versioning (MAJOR.MINOR.PATCH) and a Keep a Changelog‑style format.

## [1.3.0] - 2025-10-11
- Add per-band BPM labels (L/R low/mid/high) next to dB values.
- Add per-target tempo parameters (alpha, delta, min, hold) with optional auto-threshold.
- UI: target selector and auto toggle for tempo.

## [1.2.0] - 2025-10-11
- Per-target tempo params backend and UI wiring (alpha/delta/min/hold) with optional auto-threshold computation.

## [1.1.1] - 2025-10-11
- Fix: import `QCheckBox` for tempo auto toggle in advanced UI.

## [1.1.0] - 2025-10-11
- Add `vu_meter_120fps.py` advanced UI:
  - Adjustable dB range and FPS selector (30/60/120/240)
  - 6-band L/R meters (low/mid/high) with per-band dB display
  - Tempo lights per channel/band, L/R BPM estimate
  - Rotating file logger; NaN/Inf‑safe audio processing
  - README updated

## [1.0.0] - 2025-10-11
- Initial GUI with basic stereo VU, peak display, device selection.
## [1.4.0] - 2025-10-11
- Enable auto beat detection by default on all channels/bands (L, R, Llow/Lmid/Lhigh, Rlow/Rmid/Rhigh) with tuned parameters (alpha=0.15, k=0.8, min=0.20s, hold=0.10s).

