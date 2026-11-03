# Detection

Scythe grabs a camera frame every 30 seconds while a print runs and checks it for failures. You choose how it checks in **Settings → Detection**.

| Failure | Local model | Decision model |
| --- | :-: | :-: |
| Spaghetti | ✓ | ✓ |
| Stringing | | ✓ |
| Under-extrusion | | ✓ |
| Warping | | ✓ |
| Layer shift | | ✓ (tested on real prints) |
| Layer cracking | | ✓ |
| Part knocked loose | | ✓ |

## Local model

A YOLOv8 model trained on spaghetti, run with ONNX Runtime on the Scythe machine. It's free and nothing leaves your network. A Raspberry Pi 4 takes about 7 seconds per frame, which fits easily in the 30-second capture interval.

**Minimum confidence** (default 0.6) is the score a box needs to count. Raise it if you get false alarms, lower it if Scythe misses failures. With `log_level` set to `DEBUG` the log shows every box the model saw, including ones below the threshold.

## Decision model

Scythe sends each frame to a [decision model](https://openrouter.ai/docs/guides/community/jev) and asks two yes/no questions: does the print show any of the failures above (ignoring nozzle blobs, purge lines, brims and supports), and does it have a layer shift. Each answer is a probability. Scythe flags a failure when either one reaches the alert threshold, and the alert names the one that fired (*Layer shift detected!* or *Print failure detected!*) with both scores.

The questions and image size match the evaluation in [PRINT_FAILURE_EVAL_FINDINGS.md](../PRINT_FAILURE_EVAL_FINDINGS.md). Things that mattered in testing:

- **Crop to the part.** This mattered more than the choice of model; small failures get lost in a full frame. Drag a box on the frame in Settings.
- **Model.** The default, Liquid d1-ultra, caught almost every layer-shifted frame in good light and was the most reliable in the dark. Jev 1.14 is as good in the light at about twice the price.
- **Threshold** (default 0.6). Models aren't calibrated alike, so check it against your own prints. `DEBUG` logging records every score.
- **Light the bed.** No model was reliable on dark frames.

**Cost.** Settings shows the expected cost per print-hour for the chosen model and rates. With d1-ultra and a frame every 30 seconds it's about half a cent per hour. Nothing is sent while the printer is idle.

**Self-hosting.** Pick the *Self-hosted (Ollama)* provider to use a model served by [Ollama](https://ollama.com). The default endpoint is `http://localhost:11434/v1/systemone` and needs no key. `d3:4b` did well in testing on a desktop GPU.

## Rates

- **Capture a frame every** sets how often the camera picture refreshes (minimum 5 seconds).
- **Run detection every** sets how often a frame gets checked. `0` checks every captured frame. Use a longer interval to cut decision-model costs while keeping the dashboard picture fresh.

## Detect while idle

The **Detect while idle** switch in the top bar runs detection when no print is running, which helps when trying a new crop or threshold. It never pauses the printer or sends Discord alerts, and it turns off when Scythe restarts.
