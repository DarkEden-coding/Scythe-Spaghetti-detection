# Pipeline contracts

These are the interfaces between pipeline stages. Each stage is built and
tested independently against this file. If a stage needs to change a
contract, it updates this file in the same change and says so in its
report. Design rationale is in `DESIGN.md`.

## 1. Locations and environments

| What | Where |
|---|---|
| Data root | `/mnt/Primary/scythe-synth` (env `SCYTHE_SYNTH_ROOT`); symlinked as `datasets/synth` in the repo |
| STL corpus | `$ROOT/corpus/` (`index.json` + `stl/<id>.stl`) |
| G-code + timelines | `$ROOT/gcode_cache/<printer_profile>/<model_id>/<slice_hash>/{model.gcode,timeline.npz,slice.json}` |
| Runs | `$ROOT/runs/<run_name>/{manifest.sqlite, jobs/<job_id>/...}` |
| Pipeline package | `synthetic_data/pipeline/synthpipe/` (Python 3.12, `uv run synthpipe <cmd>`) |
| Blender code | `synthetic_data/blender/` (Blender 5.2.1 bundled Python 3.13 with numpy; no extra packages) |
| Blender binary | `~/.local/opt/blender-5.2.1-linux-x64/blender` (`synthpipe.paths.BLENDER_BIN`) |
| Slicer | OrcaSlicer 2.4.2 Flatpak `com.orcaslicer.OrcaSlicer` (needs `--filesystem=/mnt/Primary`) |

Code running inside Blender must use only the standard library and numpy,
and must not import `synthpipe`. Shared data crosses the boundary as JSON
and `.npz` files only.

Units: G-code and timelines use **mm and seconds in printer coordinates**.
Blender uses **metres**, with the plate's top-surface centre as the
bed-local origin (see `scenegen/printers.py`).

## 2. Printer profiles ↔ Blender setups

| printer_profile | Blender setup | Plate (mm) | Plate centre in G-code coords (mm) |
|---|---|---|---|
| `bambu_x1c` | `enclosed_corexy` | 256 × 256 | (128, 128) |
| `prusa_mk4` | `prusa_desk` | 250 × 210 | (125, 105) |
| `voron24_300` | `voron_topcam` | 300 × 300 | (150, 150) |
| `bambu_a1mini` | `lack_ir_night` | 180 × 180 | (90, 90) |

`ender_nozzlecam` is excluded from sequences (it has a moving camera).

The exact centre is whatever the slicer profile uses; the timeline's `meta`
records it, and Blender must read it from there, not from this table.

## 2a. Slice cache (`slice.json`, written by `synthpipe.slicing`)

Each slice directory contains `model.gcode`, `slice.json`, resolved
`machine.json`, `process.json`, `filament.json`, and `slicer.log`.
`slice_model(model_entry, printer_profile, settings) -> Path` returns the
G-code path; `pool(printer_profile) -> list[dict]` returns successful records.

`slice.json` fields: `schema_version` (1), `slice_hash`, `model_id`, `model`
(the corpus entry, including `stl` and `license`), `printer_profile`,
`settings`, `slicer` (name and version), `slicer_version`, `success`, `gcode`
(absolute path), `slice_seconds`, `estimated_print_time_s` (header estimate,
null if unavailable), `bed_size_mm`, `plate_center_mm`, `gcode_flavor`,
`nozzle_diameter`, `filament_diameter`, `machine_acceleration_mm_s2`,
`stl_sha256`, `command`, and `orca_result` (native `result.json`, including
`return_code` and `error_string`). Success requires Orca exit status zero,
`orca_result.return_code == 0`, exactly one exported plate, and all required
comments. Every Orca invocation runs with a working directory on the data
disk. Successful records include `comment_markers`
(feature/width/height/layer booleans) and `arc_moves` (count); failed records
include `error` and are also appended to `$ROOT/gcode_cache/failed.jsonl`.
The cache key covers settings, STL content, model license, flattened presets,
slicer version, and slice implementation version. Scaling is applied to a
temporary STL before Orca loads it (avoids a 2.4.2 CLI enlargement crash);
copies use `--clone-objects`. Only successful slices
with an existing G-code file are reused. Publication is atomic and locked.

`settings` contains `layer_height` (mm), `infill_pattern`, `infill_density`
(fraction 0.05–0.40), `supports` (bool), `support_type` (`normal` or `tree`),
`brim` (bool), `scale` (uniform multiplier of the original STL), `copies`
(1–4), and `wall_loops` (2–4). The caller samples scale; the pool CLI fits
25–85% of the shorter plate dimension, reduces occupancy for multiple
copies, and caps model height at 100 mm. Orca arranges copies on the plate.

G-code preserves native feature and layer comments. For Bambu output,
equivalent `;WIDTH:`, `;HEIGHT:`, and `;Z:` aliases are added alongside native
`; LINE_WIDTH:`, `; LAYER_HEIGHT:`, and `; Z_HEIGHT:` comments. Moves are
unchanged. Toolpath arc fitting is disabled; **Bambu system startup/wiping
macros still contain G2 arcs**, including full circles and relative moves.
Timeline readers must support these or explicitly exclude startup macros.
Voron retains `PRINT_START`/`PRINT_END` firmware macros; their internally
executed moves are not present in exported G-code.

## 3. Timeline `.npz` (written by `synthpipe.gcode`, read by Blender)

One row per G-code move, including travels, in execution order. `N` is the
number of moves, `L` the number of layers.

| key | dtype / shape | meaning |
|---|---|---|
| `p0`, `p1` | float32 (N,3) | start/end nozzle position, mm, printer coords |
| `t0`, `t1` | float64 (N,) | start/end time, s since print start (first extrusion of the print, after heating/homing) |
| `e` | float32 (N,) | filament length extruded by the move, mm (0 for travel; retractions are negative and are stored on zero-length moves) |
| `width` | float32 (N,) | extrusion width, mm (from `;WIDTH:` comments or derived from E) |
| `height` | float32 (N,) | layer height, mm |
| `kind` | uint8 (N,) | 0 travel, 1 outer wall, 2 inner wall, 3 sparse infill, 4 solid infill, 5 top/bottom surface, 6 support, 7 support interface, 8 skirt/brim, 9 purge/prime/wipe, 10 other extrusion |
| `obj` | int16 (N,) | normalized object-table index 0..K-1; -1 outside labelled objects |
| `layer` | int32 (N,) | layer index (0-based); -1 before the first layer |
| `layer_z` | float32 (L,) | print Z of each layer, mm |
| `layer_t0` | float64 (L,) | time the layer starts, s |
| `meta` | 0-d str (JSON) | see below |

`meta` JSON fields:

- `printer_profile`, `bed_size_mm` [x,y], `plate_center_mm` [x,y]
- `nozzle_diameter`, `filament_diameter`, `total_time_s`, `total_e_mm`
- `object_bbox_mm` [[xmin,ymin,zmin],[xmax,ymax,zmax]], computed from
  labelled nonzero-length positive-E moves when labels exist; otherwise from
  kinds 1–8 (falling back to non-purge extrusion for unannotated files).
  Startup purge/calibration geometry is excluded.
- `slicer` (name+version), `source_gcode` (path)
- `objects`: list of `{id, name, center_mm, polygon_mm, bbox_mm, n_moves}`.
  `id` is the normalized table index, assigned in first definition/start order;
  `name` is the label name (or source numeric id as a string). `center_mm`
  is [x,y] from Klipper `EXCLUDE_OBJECT_DEFINE`, otherwise the extrusion
  bounding-box centre. `polygon_mm` is the defined XY polygon or null.
  `bbox_mm` is [[xmin,ymin,zmin],[xmax,ymax,zmax]] over both endpoints of
  positive-E, nonzero-length moves assigned to this object; it is null if
  none exist, as is a computed centre. `n_moves` counts all assigned rows,
  including travel and retraction rows.

Object membership follows Prusa `M486 S`/`A`, Klipper
`EXCLUDE_OBJECT_DEFINE`/`START`/`END`, and Bambu unique-label start/stop
comments (with `M625` closing membership). Bambu `M624` alone also brackets
non-object macros and does not define an object. Outside-object moves and
purge/prime/wipe rows have `obj=-1`; skirt/brim belongs to an object only
when inside its label. Files without labels have all -1 and `objects=[]`.
Readers of older archives lacking `obj` or `objects` supply those defaults.

Both readers expose `objects`, `object_mask(obj_id)` (all matching rows), and
`segments_for_object(obj_id, kinds=None, t=None)` (row indices, optionally
filtered by kind and `t1 <= t`). The time filter includes completed rows;
partial extrusion remains available through `partial(t)` and is not included
in these indices.

Startup extrusion before the first layer marker (or first object start when
there are no layer markers), and extrusion in marked FLUSH/PURGE/WIPE blocks,
is kind 9 and does not create print layers. The first real print extrusion is
t=0; retained startup purge precedes it at negative times. Heights below 0.04 mm
fall back to the real layer step or the last valid height (initially 0.2 mm).
Sequential-object files may restart layer Z for a later copy; startup probing
and calibration Z do not enter `layer_z`.

Time model: per-move duration comes from the feedrate with trapezoidal
acceleration (use the machine's acceleration from the profile, or 5000 mm/s²
if unknown). Absolute accuracy is about ±15%; continuity matters more.

Query semantics (both sides implement the same semantics):

- **Nozzle position at time t:** find i with `t0[i] <= t < t1[i]`, then
  `p0 + (p1-p0)*(t-t0)/(t1-t0)`.
- **Extruded geometry at time t:** every move with `t1 <= t` and `e > 0`,
  plus the partial move i up to the interpolated point.
- **Current layer and Z:** `layer[i]`, `layer_z[layer[i]]`.
- **Progress:** `t / meta.total_time_s`.

## 4. Job spec (JSON, written by `synthpipe.sampler`, stored in the manifest)

```json
{
  "job_id": "s000123",
  "seed": 123,
  "generator_version": "0.3.0",
  "code_hash": "<64 hex SHA-256>",
  "group_id": "stl:<64 hex STL hash>",
  "split": "train",
  "setup": "prusa_desk",
  "printer_profile": "prusa_mk4",
  "model": {"id": "thingi10k_37266", "stl": "/abs/path.stl", "license": "CC-BY-4.0"},
  "slice": {"layer_height": 0.2, "infill_pattern": "grid", "infill_density": 0.15,
            "supports": false, "brim": false, "copies": 1, "scale": 1.0,
            "gcode": "/abs/model.gcode", "timeline": "/abs/timeline.npz"},
  "filament": {"name": "orange PLA", "rgb": [0.8, 0.2, 0.03]},
  "scene": {"variation_seed": 991, "camera_jitter": {"pos_m": 0.015, "rot_deg": 2.0},
            "light_scale": 1.0, "clutter": true},
  "failure": null,
  "keyframes": [612.0, 852.0, 902.0, 912.0],
  "windows": [{"t": 912.0, "history": [902.0, 852.0, 612.0]}],
  "events": [],
  "isp_variants": [{"profile": "webcam", "seed": 5512}],
  "render": {"samples": 48, "resolution": [640, 360], "motion_blur_shutter": 0.5}
}
```

`failure`, when present:

```json
{"type": "spaghetti" | "detached" | "layer_shift" | "nozzle_blob",
 "onset_s": 700.0,
 "params": { ...type-specific, all randomness pre-sampled here... }}
```

- `detached` covers sliding, tipping and knock-off, delamination at a
  layer (the upper section lifts), and the air printing that follows.
- `params` must fully determine the failure, so Blender re-renders the
  same thing for the same spec.
- The sampler may put only `{"seed": int}` (plus any coarse knobs, e.g.
  `{"mode": "delaminate"}`) in `params`. Blender then derives every other
  value deterministically from that seed, with its own sampling code
  (`scenegen.failures`). The same applies to `events[].params`.

Every spec records `generator_version` from `synthpipe.version` and
`code_hash()`: SHA-256 of sorted repository-relative paths and contents of
`synthpipe/**/*.py`, every file under `synthpipe/profiles/`, and
`synthetic_data/blender/**/*.py`, excluding `__pycache__`. Paths are relative
to the repository root; each UTF-8 path and file content is prefixed with its
8-byte big-endian byte length. `code_hash(short=True)` returns the first 16
hex digits; specs store the full 64 digits.

`group_id` is the canonical (lexicographically smallest) token in a connected
component joining `thing:<Thingiverse thing_id>` and `stl:<SHA-256 of STL
bytes>`. The corpus index supplies missing source metadata. Thus all files
from one Thing and byte-identical STLs across model ids stay together;
missing Thing ids fall back to STL content. Hash `group_id` with SHA-256,
interpret as a big-endian integer modulo 100: 0–79 is `train`, 80–89 `val`,
90–99 `test`. Grouping uses the full corpus plus the available slice pool.
The optional run argument `--holdout-setup NAME` overrides that setup's specs
to `test_setup`, preserving their natural setup. Default: no setup holdout.
This explicit setup override is separate from the model-group split and can
place a model group in both a natural split and `test_setup`.

Temporal sampling uses the same latent schedule for clean and failure jobs:
anchor from a 10% uniform first-180-seconds component plus 90% uniform over
the print. Keep a pre-onset window and one (70%) or two (30%) post-onset
windows in both classes (2 or 3 windows total). Latent onset is at least
20 s and 1–30 s after the anchor. Each post delta is 40% uniform 10–90 s,
60% log-uniform 90–1500 s. Reject the whole candidate if any post window
exceeds `total_time_s - 5 s`; never clip onto print end. Prints too short
for these constraints use one clean window. Failure kinds are conditioned
on early progress to emphasize first-layer detachment without changing the
shared time distribution. All history times use the same rules below.

Pool loading skips `$ROOT/gcode_cache/blacklist.jsonl` records matching a
resolved `timeline` path or `slice_hash` (one JSON object per line, with a
`reason`). It also skips timelines with more than 8,000,000 moves, inspecting
the `t0.npy` array header without decompressing its data (metadata fallback
when that array is absent).

`events` items are `{"type": ..., "t0": s, "t1": s, "params": {...}}`.

Render-side types (handled by Blender):

- `lights_off`: room or chamber lights off during `[t0, t1)`, keeping the
  camera's own IR or LED.
- `lights_on`: the sequence starts with the lights off; they are on during
  `[t0, t1)`.
- `daylight_shift`: daylight intensity or colour ramps over `[t0, t1)` and
  stays changed afterwards.
- `hand`: an arm or hand reaches in and out over `[t0, t1)` (enclosures also
  get their door opened).
- `door_open`: door or panel open during `[t0, t1)`.
- `part_removed`: plate cleared at `t0` (empty plate, toolhead parked) and the
  same print restarts at `t1`; telemetry then uses print time `t - t1`.
  Clean-sequence hard negative.

Post-side types (handled by `synthpipe.isp`):

- `ae_jump`: auto-exposure step.
- `wb_shift`: white-balance drift.
- `bump`: small camera shift or rotation, as a homography that persists
  after `t0`.

Events are active on **`[t0, t1)`** on both render and post sides, and every
sampled event contains at least one selected keyframe. Anchor `ae_jump`
within 0–3 s before a keyframe, and `wb_shift`/`bump` within 0–8 s before
one. Other events begin up to 4 s before a selected keyframe. Durations:
hand 5–60 s; door_open 20–300 s; lights_off/on 60–3600 s; daylight_shift
120–1800 s; AE jump 3–8 s; WB shift 10–120 s. Bumps persist from `t0`
through print end. Events can outlast the final sampled frame or print.
`part_removed` occurs independently in about 5% of clean sequences: the
plate is empty from `t0` until restart at `t1 = t0 + U(60,600) s`.
All other event rates and sampling rules are independent of the failure
flag. AE adaptation may decay inside its active interval; daylight ramps
and camera bumps retain their changed state afterwards as described above.

Rules:

- `keyframes` is sorted, unique, and in seconds since print start.
- Every window time and every history time is in `keyframes`.
- History entries are `t-10`, `t-60` and `t-300`, each rounded to the
  nearest keyframe. They are omitted when before t=0 (the model masks
  missing history).

## 5. Render outputs (written by Blender `scenegen.sequence.render_sequence`)

```
jobs/<job_id>/raw/
  k000.exr ...        multilayer EXR, half float, one per keyframe (index = position in spec.keyframes)
                      layers: "Combined" (linear scene-referred RGBA, NO compositor/ISP),
                              "IndexOB" (float object pass index), "Depth" (metres)
  render_meta.json
```

Each `k###.exr` layer must be readable by OpenCV (`cv2.IMREAD_UNCHANGED` with
`OPENCV_IO_ENABLE_OPENEXR=1`), or the writer documents the channel names.
Blender 5.2 writes a ZIP-compressed multipart file with channels
`ViewLayer.Combined.{R,G,B,A}` (half), `ViewLayer.Object Index.X` (float,
this is IndexOB) and `ViewLayer.Depth.Z` (float), plus
`ViewLayer.Noisy Image.*`, which Cycles always adds when denoising and
readers should ignore.

`render_meta.json`:

```json
{"job_id": "...", "blender_version": "5.2.1", "render_seconds": 41.2,
 "camera": {"profile": "webcam", "resolution": [640,360], "fov_deg": 66.0, "fisheye": false},
 "keyframes": [
   {"idx": 0, "t": 612.0,
    "telemetry": {"z_mm": 4.2, "layer": 20, "progress": 0.18, "elapsed_s": 612.0},
    "nozzle_bed_mm": [10.5, -3.2, 4.4],
    "failure_state": {"active": false, "severity": 0.0},
    "class_pixels": {"1": 81234, "2": 30211, "3": 5120, "4": 0},
    "events_active": []}
 ],
 "alignment_ok": true,
 "alignment": {"ok": true, "fit": [0.97], "labelled": true},
 "timing": {"build_s": 0.4, "keyframes": [{"update_s": 0.02, "render_s": 0.6, ...}]}}
```

- `failure_state` always has `active` and `severity` (0..1); with a failure it
  also has `type`, `visible` (active AND the failure's own class has pixels
  in this keyframe's IndexOB: 4 spaghetti, 5 detached, 6 layer_shift,
  7 nozzle_blob), `visible_px` (that pixel count), `expected_visible` (the
  generator's expectation, e.g. enough filament extruded; false negatives
  here mean occlusion) and per-type extras: `mode` and `post` (detached),
  `extruded_since_onset_mm` (spaghetti, detached), `shift_layer_printed` and
  `shift_mm` (layer_shift).
- Layer shift: the shifted geometry is class 3 while the first shifted layer
  is still printing and class 6 (everything above the shift) once it is
  complete, so class 6 pixels only exist when `shift_layer_printed`.
- Determinism: failure and scene geometry are a function of the spec and t
  only (not of which keyframes are rendered); re-rendering a spec, or a
  subset of its keyframes, gives bit-identical IndexOB/Depth and Combined
  within GPU-denoiser noise (<1% relative, at most 1 sRGB code).
- IndexOB comes from Cycles' first sample. Where the camera looks through a
  see-through panel (glass, acrylic), a sample that takes the panel's
  reflection lobe records the reflected object, so up to ~0.5% of pixels are
  isolated "pepper" (mostly class 1 inside 0 or 2). Mask writers may replace
  a pixel whose four neighbours agree on another class with that class.
- `alignment_ok` is false when the STL body does not match the toolpath
  (per-object fit below 0.5). The sampler should drop or blacklist such
  slices; the frames are still written.

`IndexOB` uses the pass indices below. Every object must have one, and 0 is
reserved for background or world.

| id | class | notes |
|---|---|---|
| 0 | background | world / nothing |
| 1 | environment | room, enclosure, clutter, loose filament scraps (hard negatives) |
| 2 | printer | frame, toolhead, bed |
| 3 | print | the intact part(s) |
| 4 | spaghetti | failure strands |
| 5 | detached | moved, tipped or delaminated part section |
| 6 | shifted | geometry above a layer shift |
| 7 | nozzle_blob | blob on the hotend |
| 8 | occluder | hand/arm, open door |

## 6. Post-processed outputs (written by `synthpipe.postprocess`)

```
jobs/<job_id>/
  frames/k000_v0.jpg ...   one JPEG per keyframe per ISP variant
  masks/k000_v0.png ...    uint8 class-ID mask, warped identically to the frame
  labels.json
```

`raw/` is deleted after successful post-processing unless the run keeps raw
output (`keep_raw`; on for a small calibration subset).

`labels.json`:

```json
{"job_id": "...", "generator_version": "0.1.0", "spec": { ...full spec... },
 "render_seconds": 41.2, "blender_version": "5.2.1", "camera": {...},
 "alignment_ok": true, "alignment": {...}, "timing": {...},
 "keyframes": [
   {"idx": 0, "t": 612.0, "telemetry": {...}, "failure_state": {...},
    "class_pixels": {"1": 81234, ...},
    "alignment_ok": true, "alignment": {...}, "timing": {...},
    "presence": {"spaghetti": 0, "detached": 0, "layer_shift": 0, "nozzle_blob": 0},
    "ignore": {"spaghetti": false, ...}, "severity": {"spaghetti": 0.0, ...},
    "area_frac": {"spaghetti": 0.0, ...}, "boxes": {"spaghetti": [], ...},
    "variants": [{"presence": {...}, "ignore": {...}, "severity": {...},
                  "area_frac": {...}, "boxes": {...}, "class_pixels": {...}}],
    "events_active": [], "render_events_active": []}],
 "variants": [{"var": 0, "profile": "webcam", "seed": 5512,
               "params": { ...resolved ISP params... }}],
 "windows": [{"t": 912.0, "history": [902.0, 852.0, 612.0]}]}
```

- Every stored frame and mask has the **source render resolution**, without
  output upscaling. Variant parameters record this as `resolution` and retain
  the profile's sensor dimensions as `native_resolution`. Different native
  aspects use a native-aspect sensor crop followed by a uniformly scaled
  cover crop to the render aspect (equivalent to an isotropic zoom), with a
  small, sequence-seeded offset. Geometry never stretches the aspect ratio.
  Stream resampling is internal; it does not change stored dimensions.
- `keyframes[i].variants[v]` describes exactly the warped mask saved for
  variant `v`, including lens distortion, crop and persistent camera bumps.
  Masks use nearest-neighbor interpolation; no class speckle cleanup is done.
- **Presence:** visible class pixels must reach
  `max(40 * output_pixels / render_pixels, 0.0002 * output_pixels)`.
  The 40-pixel minimum is defined at the source render resolution. There is
  no generator-state fallback or post-side layer-shift exception: pixels
  decide. Blender assigns the incomplete shifted layer to class 3 until it
  completes, then shifted geometry can contribute class 6 pixels.
- `ignore[type]` is true only for the job's own failure type when the
  generator reports it active, time is at/after onset, and visible pixels
  fall below threshold. Presence stays 0 in that case; downstream losses
  must mask this target. Before onset, ignore is always false.
- `severity[type]` is the generator's severity clamped to [0,1] only for
  the job's own failure type when present in that variant; otherwise 0.
  `area_frac[type]` independently records visible class pixels / frame pixels
  for every failure type, including those below the presence threshold.
- Boxes are normalized [x0,y0,x1,y1] in [0,1], with exclusive upper bounds,
  and emitted only for present types. A dilation with diameter approximately
  1.5% of frame width groups nearby fragments before connected components;
  boxes bound the original class pixels, not the dilation. Discard boxes
  with area below `(0.003 * width)^2`; retain at most eight per type,
  ranked by original class-pixel count. This grouping never changes masks.
- Keyframe-level `presence`, `ignore`, `severity`, `area_frac`, and `boxes`
  are copies of **variant 0**, for backward compatibility; they are not an
  any-variant union. Keyframe `class_pixels` retains render-resolution counts
  from metadata; each variant has its own warped output counts (string IDs).
- All keyframe render metadata, including every `failure_state` key,
  `telemetry`, `nozzle_bed_mm`, and render counts, survives raw deletion.
  `render_events_active` preserves the renderer's `events_active` verbatim;
  `events_active` is the full spec event list filtered with post-side interval
  semantics. Alignment fields fall back to job-level metadata, and keyframe
  `timing` falls back to the corresponding entry of render `timing.keyframes`.
  Job-level `render_seconds`, `blender_version`, `camera`, `alignment` and
  full `timing` are also retained.
- Active-event membership consistently uses `[t0, t1)`. In particular,
  post-side `ae_jump` and `wb_shift` cease at `t1`; a `bump` instead persists
  from `t0` onward. Render-side persistent scene effects described in §4
  may remain visible after an event's active interval.
- If the job or any keyframe reports `alignment_ok: false`, labels are still
  published with job-level `alignment_ok: false`. Post-processing appends a
  record containing `timeline` and `slice_hash` from `spec.slice`, `reason`,
  `alignment` and `job_id` to `$GCODE_CACHE/blacklist.jsonl` (default
  `/mnt/Primary/scythe-synth/gcode_cache/blacklist.jsonl`). Appends use an
  exclusive advisory lock on that file, flush and fsync; retries may append
  duplicate records. The sampler excludes blacklisted slices.

## 7. Manifest (`manifest.sqlite`, WAL mode)

`jobs` table columns:

- `job_id` TEXT PK, `spec` TEXT (JSON), `state` TEXT
- `attempts` INT, `crash_attempts` INT (default 0), `lease_owner` TEXT, `lease_expires` REAL
- `error` TEXT, `created` REAL, `updated` REAL
- `render_seconds` REAL, `post_seconds` REAL

Job states:

```
pending ──render claim──> rendering ──ok──> rendered ──post claim──> postprocessing ──ok──> done
   ^                          │                                        │
   └──── lease expired / retry┘ (attempts < 3)   failed (attempts >= 3, with error + log path)
```

Claim rules:

- Claims are a single `UPDATE ... WHERE state=? AND (lease_expires IS NULL OR lease_expires < now) ... RETURNING`.
- Workers heartbeat their lease every 30 s.
- Outputs are written to `jobs/<job_id>/.tmp_*` and renamed into place
  before the state changes. A job counts as done only once its files exist.

The Blender worker opens the same SQLite file with Python's `sqlite3`
(stdlib, available inside Blender).

Render workers atomically write `inflight/<lease_owner>.json` with `job_id` and
`owner` immediately after claiming, and remove it on completion or clean release.
Abnormal worker exit consumes one attempt for every rendering job still leased by
that owner, with `worker crashed (exit code X)` as the last error. Expired rendering
leases also consume an attempt (and increment `crash_attempts`); both paths retry
below three attempts and quarantine at three. Owner fencing prevents double
accounting or charging a newer claim. Clean SIGTERM/STOP releases consume no
attempts; postprocessing lease expiry retains its existing release behavior.
The supervisor crash budget counts distinct crashing jobs or repeated failures
before claiming a job; repeated crashes of one poison job use that job's retries.
Successful rendering progress resets consecutive crash budgets, and a clean
worker exit resets its slot's budget.

At supervisor start, `run.json` records `code_hash`, `generator_version`, and
`code_sessions` (hash, version, start timestamp). The hash comes from
`synthpipe.version.code_hash()`. A changed hash refuses resume unless explicitly
allowed with `--allow-code-change`; allowed sessions remain in `code_sessions`.
Status reports completion counts and hourly rates over the last 15 and 60 minutes,
ETA from the 15-minute rate, and total crash-attributed attempts.
New claims pause below `--min-free-gb` (default 40 GB), checking filesystem free
space and VDO physical available space for the backing device when `vdostats`
can run without sudo. Unavailable VDO statistics are silently skipped.
