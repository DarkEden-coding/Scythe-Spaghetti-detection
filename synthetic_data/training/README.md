# scythe-train

Training for the temporal print-failure model (design: `../DESIGN.md` §6).

```bash
cd synthetic_data/training
export UV_CACHE_DIR=/mnt/Primary/scythe-synth/corpus/raw/uv-cache
export HF_HOME=/mnt/Primary/scythe-synth/hf-cache
uv sync --extra dev

uv run scythe-train index                      # scan runs (incremental) + real photos
uv run scythe-train train --name dev --epochs 3
uv run scythe-train bench --batch 16 24 --frozen 6 8   # GPU step ceiling
uv run scythe-train bench --batch 1 --loader --loader-workers 4 8  # loader alone
```

- **Environment:** this is a separate uv project with torch cu128, which the
  RTX 5060 (sm_120) needs. The root `.venv` is not used.
- **Data:**
  - Samples are one per (job, window, ISP variant), taken from the `done`
    jobs of the listed runs. Each run's index is cached under
    `$SYNTH_ROOT/train_cache/index/`, so rescans only read new jobs.
  - Real photos are letterboxed into the same layout, with spaghetti boxes as
    masks. Other failure types are *unknown* there, not negative. The
    personal webcam frames are labelled clean for all types.
- **Pipeline:** workers read JPEG bytes and build mask targets. The GPU
  decodes (nvJPEG), resizes and augments whole batches. Each batch is one
  aspect bucket: 448×252 for 16:9, 448×336 for 4:3.
- **Model:**
  - DINOv2-S/14 (reg4) encoder with the first `--frozen-blocks` frozen, then
    `--fusion-blocks` time/space attention blocks, then presence and
    per-patch heatmap heads.
  - By default, encoder gradients flow through the newest frame only
    (`--history-grad false`). History frames go through the same weights
    without gradients, which roughly halves the step cost and frees the
    memory for larger batches.
- **Output:**
  - `$SYNTH_ROOT/train_runs/<name>/` holds `config.json`, `log.jsonl`,
    `eval_*.json`, `last.pt` and `best.pt`.
  - `touch <dir>/STOP` stops after the current step and saves a checkpoint.
    Re-running the same command resumes, including mid-epoch.
- **Metrics:** per-type AP, AUROC and recall at 0.1% FPR, reported
  separately for synthetic and real val. Real val only scores spaghetti.

## Sharing the GPU with the render run

Training next to `synthpipe run` takes about 7 GB of the 8 GB card by
default. Blender then fails jobs with OptiX or out-of-memory errors (main1
lost 41 jobs this way on 2026-10-18; they were requeued). Pick one:

- **Train alone (fastest):** `touch $RUN/.PAUSE_CLAIMS` to pause render
  claims, train, then remove the file.
- **Share the GPU:** `--batch-size 16 --mem-fraction 0.55`. Measured with
  batch 24 and no cap (the setting that starved Blender): training about 73
  samples/s, rendering down from about 470 to about 230–350 jobs/h. The
  capped setting has not been measured yet.

Measured throughput (RTX 5060 8 GB, 448 px, 4 frames per sample):

| Configuration | Samples/s |
|---|---|
| Eager, all 4 frames with gradients, fusion on all frames | 43 |
| + max-autotune compile | 63 |
| + gradients through the newest frame only | 92 (batch 16) |
| + fusion on the newest frame only, batch 24 | 101 (random inputs, ceiling) |
| Real loop (CPU decode, 25% 4:3 batches, losses) | 76–80 alone, ~73 next to rendering |

Things that did not help: CUDA graphs, compiling the augmentation, GPU
nvJPEG decode (its Huffman stage runs serially on the host), batch 32.
