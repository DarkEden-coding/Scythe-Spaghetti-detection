# Development

```bash
git clone https://github.com/DarkEden-coding/Scythe-Spaghetti-detection.git
cd Scythe-Spaghetti-detection
uv venv && uv pip install -r requirements-dev.txt
python -m src download-model
SCYTHE_CONFIG=/tmp/scythe.json python -m src run   # setup wizard on :8090
pytest                                              # needs no printer, model or token
ruff check src tools tests
```

To try a branch on a real printer, run the installer with `--ref`:

```bash
curl -fsSL https://raw.githubusercontent.com/DarkEden-coding/Scythe-Spaghetti-detection/master/install.sh | bash -s -- --ref my-branch
```

## Layout

- `src/monitor.py`: the capture and detect loop.
- `src/detection/`: the local ONNX detector, the decision-model client, and pricing.
- `src/notify/`: Discord and the web dashboard, both fed by the same monitor events.
- `src/web/setup_routes.py` and `src/setup_service.py`: the setup wizard's API, login, and system changes.
- `src/web/static/`: the dashboard. Plain HTML, CSS and JS with no build step.
- `install.sh`, `deploy/scythe.service`, `deploy/scythe-helper`: installation.

## Model

The runtime only needs `models/largeModel.onnx`. At 175 MB it's over GitHub's file limit, so it's an asset on the `model-v1` release, and `src/detection/model_download.py` pins its URL and SHA-256. The PyTorch checkpoint `models/largeModel.pt` stays in git for training.

```bash
python -m tools.train --data path/to/data.yaml --name my-run
python -m tools.export_onnx runs/detect/my-run/weights/best.pt
python -m tools.collect_dataset --interval 5 --output data/training
```

To ship a new model, upload the exported file to a new `model-vN` release and update `MODEL_URL` and `MODEL_SHA256`.

## Releases

Installs follow release tags. The installer picks the newest `vX.Y.Z` tag, Moonraker's update manager uses its `stable` channel, and the built-in updater moves to newer tags. To release:

1. Bump `version` in `pyproject.toml` and `src/__init__.py`.
2. Merge to `master`, then tag and push: `git tag v0.3.0 && git push origin v0.3.0`.
3. Optionally create a GitHub release from the tag with notes.

Changes to `deploy/scythe-helper` reach existing installs only when the user runs the installer again, because the installed copy is owned by root.
