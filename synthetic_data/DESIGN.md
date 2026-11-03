# Temporal print-failure detection: design

Status: draft for review (2026-10-18).

Scope: a synthetic data system and a model that watches fixed printer cameras
over time and flags spaghetti, a detached or moved part, layer shift and
nozzle blobs, with near-zero false pauses.

## 1. Decisions so far

| Topic | Decision |
|---|---|
| Capture at inference | Fixed interval, about 10 s, toolhead wherever it happens to be (no parking) |
| Failure types | Spaghetti; detached/moved part (incl. delamination, air printing); layer shift; nozzle blob. Warping and layer adhesion are out of scope for now |
| Output | Per-type confidence and a coarse location heatmap |
| History | Multi-scale: frames at t, t-10 s, t-60 s, t-300 s |
| Extra inputs | Moonraker telemetry: Z, layer, progress %, time since print start |
| Label meaning | "Failure visible in the newest frame" (presence), per type |
| Action | Per-type policy: pause on spaghetti or detachment; notify for layer shift, blob, or low confidence |
| False alarms | Very low, at most about 1 false pause per 100 print-hours |
| Inference hardware | Desktop/server GPU (no size constraint at 0.1 Hz) |
| Rollout | Mostly replace YOLO. Fine-tune YOLO on a little synthetic data first so the baseline comparison is fair |
| Prints | Real G-code, sliced from a downloaded corpus of permissively licensed STLs |
| Cameras | Fixed mounts only; the nozzle cam is dropped |
| Drift | Auto-exposure/WB, lighting changes, camera bumps, occlusion (hands, door, glare) |
| Render speed-ups | Low samples + denoise, ISP applied in Python, render at model resolution |
| Sequences | Simulate the full print timeline; render only the keyframes a window needs |
| Data balance | Clean-heavy: about 60% clean (many hard negatives), about 40% failure |
| Image encoder | Pretrained ViT/ConvNeXt (DINOv2-S as the starting point) with a temporal transformer |
| Compute | This PC (RTX 5060 8 GB) for days, with hardened pause/resume |
| Real data | Mix into training, use as the YOLO baseline, hold out about 20% as a real-only test |

## 2. What the existing real data is

The data comes from Roboflow workspace `dark-eden-nuheg`, raw exports
downloaded to `datasets/` (gitignored).

**`3d-printing-fail-detection` v10** (raw; v5–v9 are the same images with
baked-in grayscale/flip/blur augmentation):

- 1,085 images from about 630 unique sources, with 3,254 `spaghetti` boxes and
  411 `printer-gantry` boxes. Every image has at least one box, so there are no
  clean negatives.
- Mostly web-scraped phone photos of finished spaghetti disasters: Reddit,
  Facebook and Tumblr filenames, close-ups, prints taken off the printer or
  held in a hand. One phone video contributes 12 frames, the only temporal
  material.
- Split leakage: 46 source images appear in more than one split, and a coarse
  perceptual hash flags about 155 near-duplicate pairs across splits. Some
  are false positives, but the current valid/test numbers are optimistic.

**`personal_printer_augmentation` v1:**

- 493 images, but only 36 unique source frames; the rest are augmented copies,
  spread across all three splits.
- Real webcam frames from your own printer, with a magenta-ish cast and a view
  under the gantry. Only gantry boxes, so these are real clean negatives in
  the deployment domain.

What this means:

- The main set measures recall on obvious, finished spaghetti, photographed by
  phone. It cannot measure detection latency, false-alarm rate, or temporal
  behaviour on webcams.
- The personal set is the only real data in the deployment domain. Use it to
  set the camera profiles and as real negatives.
- The most valuable thing to add is real 10 s timelapses from your printers.
  Successful prints cost nothing to collect and are exactly what the
  false-pause rate has to be measured on; `tools/collect_dataset.py` already
  captures at an interval. A few dozen real failures would let us validate
  latency.
- The real splits must be rebuilt, grouped by source stem plus perceptual-hash
  cluster, before any number from them is trusted. The 20% holdout is drawn
  from these groups.

## 3. System overview

```
STL corpus ──> slicer CLI (per printer profile) ──> G-code
                                                      │
                                     G-code parser + time model
                                                      │
 job spec (seed) ──> scenario sampler ──> print timeline + failure timeline
   (setup, model, camera pose, lighting/clutter, failure type & onset,
    drift events, which keyframe times to render)
                                                      │
                     Blender worker (persistent, GPU) │ renders keyframes:
                     linear RGB (EXR half) + class mask + depth + metadata
                                                      │
              Python ISP (K camera variants per render, drift, JPEG)
                                                      │
          dataset shards (frames, masks, labels.json, manifest)
                                                      │
        temporal model training  <── real frames (single-frame windows)
                                                      │
          evaluation: synthetic held-out + real holdout + YOLO baseline
```

## 4. Generator

### 4.1 Prints from real G-code

- **Corpus:** about 500 models from Thingi10K, filtered to permissive
  licences, a single body, a printable size, and some with overhangs (so
  supports appear). Downloaded once by a script and cached.
- **Slicing:** PrusaSlicer or OrcaSlicer CLI, one profile per printer family
  (Bambu-like, MK4, Voron, cantilever). Settings are randomized: layer
  height, infill pattern and density, supports on or off, brim, number of
  copies, scale.
- **Parsing:** G-code becomes typed segments (perimeter, infill, support,
  travel, purge) with layer index, Z, width and feedrate. A time model
  integrates feedrates, with an acceleration-limited approximation, so every
  timestamp maps to an exact toolpath position, completed geometry, nozzle
  XYZ and Z.
- **Geometry at time t**, kept cheap:
  - *Body below the top few layers:* the STL bisected at that height, with
    the layer-line shader. Walls are what the camera sees.
  - *Top 1–3 layers and the partial current layer:* real toolpath tubes, so
    the exposed infill and perimeters are accurate.
  - *Supports:* tubes from the support segments, decimated. They are thin and
    visible, and they are prime false-positive bait, so they are worth the cost.
- **Telemetry** (Z, layer, progress, elapsed) comes straight from the timeline,
  exactly what Moonraker would report at the same moment.

### 4.2 Failure timelines

Every failure has an onset time and evolves with time since onset. The
newest frame's label is "present" once the failure is visible, with a
severity value. The visibility threshold is set per type, using the mask
pixel count from the camera's view.

- **Spaghetti:** the existing strand simulator, driven by how much filament
  has been extruded since onset (from the G-code), so the nest grows at the
  real extrusion rate.
- **Detached or moved part:** at onset the part, or its upper section for
  delamination, gets a rigid transform: slide, tilt, topple, or carried along
  by the nozzle. Afterwards the toolpath continues where the part used to be:
  - *Over empty space:* the part stops growing (air printing), and filament
    collects as spaghetti or as a blob on the nozzle.
  - *Over a displaced part:* the nozzle crashes into it, so it may be pushed
    again.
- **Layer shift:** at layer k, all later geometry is offset by dx/dy (one or
  more shifts), producing the visible stair-step.
- **Nozzle blob:** a growing lump around the heater block and nozzle, with
  some print deformation under it. It ends in a blob-of-death or in the part
  being knocked off.

### 4.3 Hard negatives (the about-60% clean side)

These are sampled independently of the failure flag, so no cue can be a
shortcut:

- **Filament that isn't spaghetti:** supports and tree supports, stringing,
  skirts and brims, purge lines and blobs, poop in chutes and buckets, loose
  offcuts. Most of this is in `clutter.py` already.
- **Events between frames:** a hand reaching in, the door opening, a part
  removed at the end of a print, the plate swapped, a spool runout pause with
  the toolhead parked.
- **Camera and lighting:** lights turning on or off, daylight changing, IR
  mode switching, an auto-exposure jump, a small camera bump.
- **Toolhead positions:** anywhere along the real G-code path, since capture
  isn't parked.

### 4.4 Scenes and cameras

- **Setups:** keep the four fixed-camera setups (enclosed CoreXY, Prusa desk,
  Voron top cam, Lack IR) and add more printer families over time.
- **Camera pose:** randomized within each mount's plausible envelope.
- **Per-sequence variety:** filament, plate, lighting and clutter are seeded
  per sequence.
- **Camera profiles:** calibrated against real webcam frames, starting with
  the personal-printer set's NoIR-like colour cast.
- **Drift:** bumps, AE/WB jumps and occlusion events are scheduled on the
  timeline. Lighting changes and occluders are rendered; AE/WB and small
  bumps (homography and crop) are applied in Python.

### 4.5 Rendering and throughput

- **Scene reuse:** the scene is built once per sequence; each keyframe
  updates only the print geometry, toolhead pose, failure geometry and
  scheduled events.
- **Render settings:** Cycles at 32–64 samples with OIDN denoising, at about
  640 px wide.
- **Outputs per keyframe:** linear RGB (EXR half), a class-ID mask from the
  pass indices, and depth.
- **ISP in Python** (numpy/torch port of `camera.py`): the 5 profiles plus
  randomized parameters. K = 3 variants per render triples the samples
  without re-rendering. JPEG encoding is real (PIL/libjpeg) at the
  randomized quality.
- **Rough budget, to be benchmarked:**
  - About 1.5–3 s per keyframe at 640 px and 48 samples on the RTX 5060.
  - About 6 keyframes per sequence covers 1–2 windows with full history.
  - That gives roughly 5–8k sequences per day before the ISP multiplier.
- **Storage:** EXR is kept only for a small calibration subset. Everything
  else is stored as ISP'd JPEG plus masks, with the job spec so any frame can
  be re-rendered.

### 4.6 Hardened pause/resume

- **Jobs:** a job is a deterministic spec (seed, generator version, every
  sampled parameter) written to a manifest before rendering.
- **Manifest:** SQLite (WAL) tracks job states (pending, leased, done,
  failed) with lease timestamps. Expired leases are reclaimed after a crash or
  power loss.
- **Workers:** one persistent Blender process per GPU pulls jobs in a loop.
  This avoids Blender startup and scene teardown per job, and lets several
  sequences reuse one loaded STL or G-code timeline.
- **Atomic writes:** output goes to a temp dir, gets checksummed, then is
  renamed into its shard. A job is done only after the rename. Re-running is
  always safe because completed jobs are skipped.
- **Stopping:** SIGINT/SIGTERM finishes the current keyframe, marks the job
  pending, and exits. `resume` simply restarts the workers.
- **Failures:** a job gets a retry limit, then is quarantined with its log so
  one bad STL cannot stall the run. A disk-space guard pauses the run when
  free space is low.
- **Versioning:** the generator version and a code hash are stored on every
  job, so datasets built across code changes stay traceable.

## 5. Dataset format

```
dataset/
  manifest.sqlite
  shards/000123/<sequence_id>/
    spec.json            full job spec (reproducible)
    frames/<t>.jpg       ISP'd frames, one per keyframe per ISP variant
    masks/<t>.png        class-ID mask (0 bg, 1 env, 2 printer, 3 print,
                         4 spaghetti, 5 detached/delaminated, 6 shifted
                         section, 7 nozzle blob)
    labels.json          per keyframe: timestamp, telemetry, per-type presence
                         and severity, boxes derived from masks, toolhead
                         position, drift events active
```

Training windows are assembled at load time from each sequence's keyframes,
so the same renders can feed different history configurations later.

## 6. Model

- **Inputs:** 4 frames (t, t-10, t-60, t-300 s) at 448 px, plus a telemetry
  vector (Z, layer, progress, elapsed, actual time gaps). Missing history,
  such as at print start, is masked rather than zero-filled.
- **Encoder:** pretrained DINOv2-S/14, shared across frames, fine-tuned with
  the last blocks unfrozen or with LoRA. This gives patch tokens per frame.
- **Temporal fusion:** 2–4 factorized space-time transformer blocks over
  patch tokens, with learned embeddings for the time gaps and telemetry
  tokens.
- **Heads:**
  - *Presence:* multi-label per-type logits (focal BCE).
  - *Location:* per-type patch heatmap (BCE/dice against the masks), which
    also drives the Discord overlay.
- **Single-frame fallback:** history is randomly dropped during training, so
  the same model handles one frame alone.
- **Real data in training:** real frames enter as single-frame windows with
  history masked. Spaghetti boxes become weak heatmap targets and the other
  types are left unlabelled, not negative. The 36 personal-printer frames
  enter as clean frames.
- **Trigger logic:** per-type thresholds plus persistence, e.g. pause when 2
  of the last 3 windows exceed the threshold. That adds about 10 s of
  latency for a large false-positive reduction. Thresholds are chosen on
  clean real timelines to meet the 1-per-100-hours target.

## 7. Evaluation

- **Synthetic test:** held-out STL models, held-out camera poses and seeds,
  plus one fully held-out setup to measure generalization to an unseen
  printer and camera.
- **Real test:** the rebuilt, de-duplicated 20% holdout from Roboflow, for
  spaghetti recall on real photos. The personal-printer frames measure false
  positives, and real timelapses, once collected, measure latency and the
  false-pause rate.
- **Baseline:** current YOLOv8-L as is, and again after a short fine-tune on
  single synthetic frames (boxes from masks, same epochs budget). It runs per
  frame through the same trigger logic.
- **Metrics:**
  - per-type AP and AUROC;
  - event-level detection latency after onset;
  - false pauses per 100 print-hours on clean timelines;
  - heatmap pointing accuracy.

## 8. Risks and open questions

- **Domain gap.** The real labelled data is phone photos, not webcams, so
  real recall numbers will be loose until timelapses exist. Collecting clean
  timelapses from your printers is the cheapest, highest-value action.
- **Unparked capture.** The toolhead and bed move between every frame. The
  model has to learn this from the G-code timeline; telemetry and multi-scale
  history should help.
- **Slicer dependency.** PrusaSlicer or OrcaSlicer CLI in the pipeline (an
  AppImage or Flatpak on this machine), plus per-printer profiles to maintain.
- **Corpus licensing.** Only permissive Thingi10K licences are used, and the
  licence is recorded per model in the spec.
- **Throughput is an estimate.** Benchmark first; if it is too slow, cut
  samples or resolution further, or render fewer keyframes per sequence.
- **Roboflow key.** It was pasted into chat; rotate it, and use
  `ROBOFLOW_API_KEY` from the environment from now on.

## 9. Phases

1. **Foundations:** rebuild and de-duplicate the real splits; benchmark
   render speed; set up the job manifest and worker with pause/resume.
2. **G-code prints:** corpus download, slicing, parser and time model, and
   timeline-driven print geometry in the existing setups.
3. **Failure timelines:** spaghetti (time-driven), detachment and air
   printing, layer shift, nozzle blob; hard negatives and drift events.
4. **Python ISP and dataset writer:** shards, masks, labels; a small pilot set
   of about 500 sequences, checked by eye.
5. **Model v0 and baselines:** temporal model, YOLO fine-tune baseline,
   evaluation harness.
6. **Scale up:** multi-day generation run, model v1, threshold calibration,
   integration into `src/monitor.py` alongside YOLO, then replacement.
