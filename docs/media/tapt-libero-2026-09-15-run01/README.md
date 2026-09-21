# TAPT LIBERO — 2026-09-15 — run01

Task: put both the cream cheese box and butter in the basket (`libero_10`, task 1). All 15 fixed episodes are retained, including failures.

| File | Configuration | Native result | Action steps | GPT requests |
|---|---|---|---:|---:|
| [baseline-episode000.mp4](baseline-episode000.mp4) | baseline | Success | 241 | 0 |
| [baseline-episode001.mp4](baseline-episode001.mp4) | baseline | Success | 264 | 0 |
| [baseline-episode002.mp4](baseline-episode002.mp4) | baseline | Success | 242 | 0 |
| [baseline-episode003.mp4](baseline-episode003.mp4) | baseline | Success | 253 | 0 |
| [baseline-episode004.mp4](baseline-episode004.mp4) | baseline | Success | 284 | 0 |
| [vlm-standard-episode000.mp4](vlm-standard-episode000.mp4) | vlm-standard | Success | 286 | 12 |
| [vlm-standard-episode001.mp4](vlm-standard-episode001.mp4) | vlm-standard | Success | 280 | 14 |
| [vlm-standard-episode002-failed.mp4](vlm-standard-episode002-failed.mp4) | vlm-standard | Failure | 174 | 9 |
| [vlm-standard-episode003-failed.mp4](vlm-standard-episode003-failed.mp4) | vlm-standard | Failure | 40 | 2 |
| [vlm-standard-episode004.mp4](vlm-standard-episode004.mp4) | vlm-standard | Success | 255 | 10 |
| [vlm-tapt-episode000-failed.mp4](vlm-tapt-episode000-failed.mp4) | vlm-tapt | Failure | 190 | 10 |
| [vlm-tapt-episode001-failed.mp4](vlm-tapt-episode001-failed.mp4) | vlm-tapt | Failure | 255 | 9 |
| [vlm-tapt-episode002-failed.mp4](vlm-tapt-episode002-failed.mp4) | vlm-tapt | Failure | 425 | 20 |
| [vlm-tapt-episode003-failed.mp4](vlm-tapt-episode003-failed.mp4) | vlm-tapt | Failure | 240 | 9 |
| [vlm-tapt-episode004-failed.mp4](vlm-tapt-episode004-failed.mp4) | vlm-tapt | Failure | 120 | 5 |

Initial states 0–4, environment seed 7, policy sampling seed 0 once per configuration, 20 wait steps, 520 action-step budget, execute 5 actions per prediction. Videos are original simulator RGB at 10 fps, without parameter overlays. Playback time is not compute wall time.

[Full results and limitations](../../tapt-libero-run01.md) · [SHA-256 inventory](../manifest.json). LIBERO / robosuite / MuJoCo assets retain their upstream licenses.
