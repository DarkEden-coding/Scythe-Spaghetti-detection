# Print-failure detection with decision models: findings

Written 2026-10-28 from the evaluation work done 2026-10-17 to 2026-10-28.
Harness and raw data: `/mnt/Primary/Projects/Print-Failure-Eval` (scripts, `results/*.csv`, `results/roc_data.json`, frames in `data/raw/printer/`).

## Goal

Find out whether "decision" models (Jev, Liquid d1/d3, clef-flash, pplx-decider) detect 3D-print failures well enough, at a low enough false-alarm rate, to build a monitoring system on the Scythe printer. The metric throughout is **detection rate vs false-positive rate** (ROC, AUC, and detection at a 5% false-alarm threshold, set at the 95th percentile of clean-frame scores).

## Setup

- **Printer:** Klipper/Moonraker on a Pi 4 (`scythe-mainsail`, `ssh darkeden@100.64.34.46`). The camera is a fisheye looking up from below the bed, so the part is small in the frame.
- **Capture service:** `print-frames.service` (source: `deploy/print_frames.py` in the eval repo) saves one 1024 px frame per minute, only while `print_stats` is `printing` and `print_duration` is at least 300 s. Output goes to `~/print_frames/<start>_<job>/` with `frames.csv` and `job.json` (final Moonraker status). There is a 3 GB cap. The printer powers off right after a print, so the job-end check runs before the Klippy query (fixed 2026-10-27).
- **Prints used (806 frames):**

  | Job | Date | Material | Outcome | Used as |
  |---|---|---|---|---|
  | 0000DB | Oct 17 | PC | completed, real layer shifts | positives |
  | 0000DC | Oct 21 | PLA | cancelled, no visible shift | negatives |
  | 0000DE | Oct 22 | PLA | cancelled, no visible shift | negatives |
  | 0000DF | Oct 26 | PLA | cancelled after 10 min (8 frames) | not scored |
  | 0000E0 | Oct 26 | PLA | completed, 7.73 h, no defects (user-confirmed) | negatives |

- **Image views:** the full frame downscaled to 384 or 1024 px, or a crop around the part (box (330,120,950,350) of the 1024x576 frame) at 512 px (`crop_512`) or enlarged to 1240 px (`crop_2x`).
- **Questions** (yes/no, scored as P(yes) via the `noul` type): `any_failure` (the optimized prompt from the public-dataset search) and `layer_shift` (a layer-shift-specific one).
- **Lit vs dark:** `low_light()` calls a frame dark when mean HSV saturation is below 40. The room goes dark at night, so about half the frames are dark.
- **Sample for model comparison:** every 3rd frame of four prints (Oct 17, 21, 22, 26-E0), both views. 83 shifted and 110 clean lit frames, 68 shifted and 161 clean dark frames. pplx-decider 27b used a smaller sample (every 9th frame, `crop_512` only: 28 shifted and 37 clean lit, 23 and 54 dark).

## Earlier phases (public data, then the printer)

1. **Public datasets** (14,390 images after dedup, spaghetti only 134 images from 8 prints). A one-list prompt with "ignore nozzle residue/purge/brim/supports" beat the original (AUC 0.774 vs 0.709 on the calibration half) and held up on the test half for all 9 models (+0.005 to +0.099 AUC). Downscaling to 384 px halved the token cost with almost no loss. Best on public data: pplx-decider-flash-v1.2 (AUC 0.760, about $0.003 per 1k images) and Jev 1.14 (AUC 0.750, 48% detection at 5% FP, about $0.03 per 1k).
2. **Printer frames, off the shelf.** On the two cancelled PLA prints (Oct 21, 22), no layer shift is visible at this camera's resolution, and models scored their last frames like clean frames. The 384 px full-frame downscale also hides the Oct 17 shifts. Dark frames score higher on `any_failure` than lit ones.
3. **Jev on the Oct 17 print with the enlarged crop and the `layer_shift` question.** 98% of lit shifted frames score at least 0.5, vs 0% of lit PLA frames. Dark frames of the same part drop to about 0.23 (0% flagged).
4. **Oct 26 clean print.** The job completed, 7.73 h, steady temperatures, and one expected 14-minute pause gap (15:06 to 15:20). Frames are lit until 17:22 (z up to 23 mm) and dark after. Jev `layer_shift` flagged 0 of 459 frames at 0.5 or above (max 0.45). `any_failure` was at 0.5 or above on 9% of lit and 21% of dark frames, so that question is noisy.

## Model comparison (best view and question per model)

Shifted (Oct 17) vs clean, on the shared sample. "Det@5%" is detection at the 5% false-positive threshold.

| Model | Where | Lit AUC | Lit det@5% | Dark AUC | Dark det@5% |
|---|---|---|---|---|---|
| d3 9b | local (Ollama) | 1.00 | 100% | 0.91 | 65% |
| pplx-decider 27b | local (Ollama), reduced sample | 1.00 | 100% | 0.80 | 48% |
| clef-flash 9b | local | 1.00 | 99% | 0.94 | 59% |
| Jev 1.14 | OpenRouter | 1.00 | 99% | 0.79 | 38% |
| Liquid d1-ultra | OpenRouter | 1.00 | 98% | 0.93 | 68% |
| d3 4b | local | 0.99 | 99% | 0.89 | 54% |
| d3 2b | local | 0.99 | 92% | 0.57 | 2% |
| Liquid d1.1 | OpenRouter | 0.96 | 89% | 0.73 | 10% |
| d1-omni 600m | local | 0.80 | 28% | 0.43 | 7% |
| d1 3b | local | 0.73 | 40% | 0.92 | 46% |

(Lit AUCs shown to two decimals; the values for Jev, d1-ultra and clef-flash are 0.998, 0.996 and 0.999.)

**Speed** (median seconds per `crop_512` frame, RTX 5060 8 GB): d1-omni 0.06, d3 2b 0.17, d1 3b 0.17, d3 4b 0.56, clef-flash 0.86, d3 9b 1.39. The 9b and 27b models spill partly into CPU/RAM (pplx-decider ran about 88% on CPU, roughly 4 h for the full sample, so it was run on the reduced one).

**Cost:** Jev costs about $0.005 per print-hour at one frame per minute; d1-ultra about $0.007 per print-hour. Scoring both Liquid models on the whole sample cost $0.04.

## Conclusions

- **In good light, a layer shift is easy to detect** with a part crop and a layer-shift-specific question. Several models (d3 9b, clef-flash 9b, d3 4b, Jev, d1-ultra) reach AUC 0.99 or better with detection at or above 98% at 5% false alarms.
- **In the dark, nothing is reliable.** The best dark results are d1-ultra (68% detection) and d3 9b (65%); Jev drops to 38%. The real fix is a light on the bed, not a better model.
- **Cloud choice:** Liquid d1-ultra, best dark performance and cheap. Jev is just as good in the light.
- **Local choice on this desktop:** d3 4b (fast, nearly as good as 9b in the light), or d3 9b for the best dark results at about 1.4 s per frame.
- **Raspberry Pi 4:** nothing proven. d3 2b and the d1-omni/d1 3b models are too weak. d3 4b is the smallest model that looks useful and has not been tested on the Pi. The Pi has about 7.2 GB free RAM and 5.7 GB free disk, aarch64.
- **View and question matter more than model size for several models:** the enlarged `crop_2x` plus the `layer_shift` question is usually best in the light; `crop_512` is enough for some. A 384 px full-frame downscale is too coarse.

## Caveats

- **Confounds.** The only shifted print is also the only PC filament print and the tallest. Lit clean frames from Oct 26 only cover z 0 to 23 mm, while lit shifted frames cover z 23 to 51 mm. Part of the "detection" could be the model recognising the PC part or the taller part. A second clean PC print run to full height in good light would settle this.
- **One failed print, one failure type.** Only layer shifts were tested on the printer. Spaghetti, detachment and so on are untested on this camera.
- **d1 3b reliability.** About 25% of calls returned HTTP 500 ("failed to process mtmd chunk"). Eight retries still left 44 of 844 requests failing, and those are excluded from its stats.
- **pplx-decider 27b** ran on a smaller sample (28 shifted and 37 clean lit frames), so its numbers are noisier.
- Detection thresholds were set on the same clean frames they are measured against, so det@5% is slightly optimistic.

## Next steps (not done)

- Light the bed, then rerun the dark-frame comparison.
- Print a second clean PC part to full height in good light.
- Install Ollama on the Pi in the home folder (no sudo needed) and time d3 2b and d3 4b there.
- Full-sample pplx-decider 27b run (about 4 h).
- Fix or drop the `layer` column in the capture service's `frames.csv` (ended at 513 of 878 actual layers).

## How to reproduce

```bash
cd /mnt/Primary/Projects/Print-Failure-Eval

# cloud models (needs an OpenRouter key in OPENROUTER_API_KEY)
uv run python scripts/printer_layer_shift.py --all          # Jev, all frames, crop_2x
uv run python scripts/openrouter_eval.py liquid/d1.1 ~liquid/d1-ultra-latest

# local models: Ollama v0.40.11 unpacked at ~/.local/ollama, models in /mnt/Primary/ollama-models
OLLAMA_MODELS=/mnt/Primary/ollama-models OLLAMA_HOST=127.0.0.1:11434 ~/.local/ollama/bin/ollama serve
PYTHONPATH=scripts uv run python scripts/ollama_eval.py d3:4b d3:9b clef-flash:9b
STEP=9 VIEWS=crop_512 PYTHONPATH=scripts uv run python scripts/ollama_eval.py pplx-decider:27b

# ROC data and summary table
uv run python scripts/roc_data.py        # writes results/roc_data.json
```

The Ollama endpoint is `POST /v1/systemone` with `{model, state, images, questions}` and returns `answers[q].noul`. OpenRouter decision models are at `POST https://openrouter.ai/api/alpha/decisions`; the public `/models` list omits them, so probe the endpoint directly (`liquid/d1.1` resolves to d1.1-20261027, `~liquid/d1-ultra-latest` to d1-ultra-20261015).
