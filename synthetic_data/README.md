# Synthetic data

Blender scenes that render 3D printers mid-print as seen through cheap
cameras, with optional print failures. Two entry points:

- `render_job.py` renders a **sequence job** (CONTRACTS.md §4/§5): real G-code
  timeline, failure timelines, render-side events, linear EXRs plus masks.
  This is what the pipeline worker calls (`scenegen.sequence.render_sequence`).
- `build_scene.py` builds a single procedural still (the original showcase
  scenes, with the compositor ISP for preview JPEGs).

```
blender/render_job.py         CLI: render job spec(s) to a directory (+ PNG previews, sheets)
blender/build_scene.py        entry point: builds one scene, optionally saves/renders it
blender/tools/make_test_spec.py  write a test job spec for a timeline + STL (numpy only)
blender/scenegen/
    sequence.py               render_sequence(spec, out_dir, progress): build once, render keyframes
    gcodeprint.py             print geometry at time t from the timeline + STL
    failures.py               spaghetti / detached / layer_shift / nozzle_blob timelines
    events.py                 lights, daylight, hand, door, part removal; light registry
    meshgen.py                numpy mesh builders: beads, tubes, blobs, STL slab clipping
    imageio.py                EXR reader (multipart, ZIP) and PNG writer
    sheet.py                  contact sheets of rendered jobs
    timeline.py               timeline .npz reader (owned by the G-code stage)
    setups.py                 named setups: surroundings + printer + lights + camera mount
    printers.py               printer models and toolheads, each exposing a motion Rig
    printpart.py              procedural parts for build_scene.py, delamination + spaghetti
    clutter.py                props and debris: purge blobs, offcuts, tools, cables, spools
    camera.py                 lens/sensor model, render settings, compositor ISP profiles
    materials.py              material factory driven by a per-setup palette
    util.py                   geometry primitives, collections, pass indices
blender/scenes/<setup>.blend  saved example of each setup (matches renders/)
renders/<setup>.jpg           example frame from each setup's camera
```

## Building / rendering

Blender 5.2 is at `~/.local/opt/blender-5.2.1-linux-x64/blender`.

```bash
BLENDER=~/.local/opt/blender-5.2.1-linux-x64/blender
$BLENDER -b -P synthetic_data/blender/build_scene.py -- \
    --setup voron_topcam --seed 5 --failure spaghetti --filament "yellow PLA" \
    --save synthetic_data/blender/scenes/voron_topcam.blend \
    --render synthetic_data/renders/voron_topcam.jpg
```

`--seed` changes the parts on the plate (1-3 brackets or round bosses, with
varying size, placement, rotation and infill), the print height, how far
through the current layer the toolhead is, the failure details, and the
sensor-noise pattern. It also picks a filament unless `--filament` sets one.
Rendering through the script uses the GPU (OptiX/CUDA) and takes about
5-10 s per frame at 256 samples. Opening a `.blend` directly renders on
whatever device Blender's preferences select.

## Sequence jobs

```bash
BLENDER=~/.local/opt/blender-5.2.1-linux-x64/blender
python3 synthetic_data/blender/tools/make_test_spec.py --timeline .../timeline.npz \
    --stl .../input.stl --failure detached --mode tip --out /tmp/spec.json
$BLENDER -b -P synthetic_data/blender/render_job.py -- --spec /tmp/spec.json \
    --out /tmp/job --png-preview [--sheet /tmp/sheets] [--closeup]
```

`render_sequence` builds the scene once per job (setup, palette, clutter and
camera jitter from `scene.variation_seed`), then for each keyframe swaps the
print/failure meshes, re-keys the toolhead (frames 0..2 around frame 1, so
Cycles motion blur follows the timeline's velocity), applies events and
renders `k###.exr` (no compositor). It reads each EXR back to count IndexOB
pixels per class into `render_meta.json`. `--png-preview` adds plain sRGB
`k###.png` and false-colour `k###_mask.png`; `--closeup` swaps in a debug
camera on the part. The scene is cleared afterwards, so one Blender process
can run job after job.

**Print geometry** (`gcodeprint.py`). Timeline mm become bed-local metres
around `meta.plate_center_mm`. Each copy is one slicer object label (falling
back to footprint islands for unlabelled timelines); the STL (the slice
cache's scaled `input.stl` when present) is placed at the object's wall box
and rotated by whatever angle best matches its outer walls, since Orca's
arrange rotates parts. At time t a copy is: the STL clipped below the top of
the layer three layers under the nozzle, capped with a dark "deep infill"
material; every extrusion above that, including the partial move, as
flattened 6-sided beads; supports for all completed layers (collinear runs
merged, every k-th layer with thicker beads above 60k segments); and brim,
skirt and purge lines. `alignment_ok` in render_meta flags slices whose STL
does not match the toolpath.

**Failures** (`failures.py`), all drawn from `failure.params.seed`; one object
fails (chosen by seed among those printing at onset), the others keep going:

| type | at onset | afterwards | classes |
|---|---|---|---|
| spaghetti | the object stops growing | extrusion over it falls as strands that drift, coil and pile up on a heightfield; 40% also grow a nozzle blob | 3, 4, (7) |
| detached `slide` | object slides 6-35 mm and yaws | air printing: strands and/or a blob | 5, 4, 7 |
| detached `tip` | leans 8-25 deg or topples about a base edge | same | 5, 4, 7 |
| detached `knockoff` | lands elsewhere on the plate, often on its side | same | 5, 4, 7 |
| detached `delaminate` | the section above a crack layer tilts 3-12 deg | strands drape over it | 3, 5, 4, 7 |
| layer_shift | from the next layer, everything (and the nozzle) offset 1-8 mm, 30% shift twice | stair-step | 3, 6 |
| nozzle_blob | a lump starts at the tip | grows around the heater block over 4-20 min; layers printed meanwhile are ragged and under-extruded | 3, 7 |

The strand nest is simulated once, up to 1800 s after onset whatever the
keyframes (0.6 mm samples for the first 18 m of path, coarser after), and
revealed by deposition time, so the nest at T-300, T-60, T-10 and T is one
consistent history and does not change when only some keyframes are
rendered. Detached parts are placed by drawing candidate motions from the
seed until the moved part lands clear of the other copies' footprints and on
the plate (a toppled part pushed into a neighbour would be hidden inside it);
a topple with no clear side becomes a lean. Layer-shift geometry stays class
3 until the first shifted layer completes, then everything above the shift
is class 6. Severity is filament extruded since onset (spaghetti),
displacement (detached), shift size, or blob growth. In render_meta,
`failure_state.visible` is measured: the failure's class has IndexOB pixels
(`visible_px`); the generator's guess is `expected_visible`.

Renders are reproducible: same spec, same masks bit for bit (and the same
for a subset of the keyframes); Combined differs only by GPU OIDN noise
(<1% relative). `mathutils.noise.noise_vector` is seeded per process, so
scenegen uses `util.noise_vector` (scalar Perlin) instead.

**Events** (`events.py`): see CONTRACTS.md §4. "Main" lights per setup are the
chamber LED, ceiling lamp and Voron LED strip; the Lack IR illuminator and
toolhead LEDs are never switched. Glass and acrylic are thin see-through
panels, so masks show what is behind them.

**Speed** (RTX 5060, 640x360 or 640x480, 48 spp, OptiX, GPU OIDN, light tree
off): build 0.2-6.6 s, per-keyframe update 0.01-1.1 s, render 0.4-1.7 s. A 19.6M-move
timeline (two 13x-scaled gyroid copies) builds in 15 s with 4 GB peak RSS;
live toolpath beads are capped at 250k segments and supports at 60k.

## Setups

| setup | printer | camera | look |
|---|---|---|---|
| `enclosed_corexy` | Bambu-style enclosed CoreXY, textured gold PEI | built-in chamber cam, top front-left | cool LED, cyan cast, LED-PWM banding |
| `prusa_desk` | Prusa MK4 bed slinger, satin black sheet | USB webcam on a desk tripod | warm room plus window daylight, cleanest image |
| `voron_topcam` | Voron 2.4, flying gantry, smoked panels | Pi NoIR fisheye, top front-right | magenta cast (no IR filter), heavy JPEG |
| `ender_nozzlecam` | Ender-3, silver V-slot, glass bed with clips | USB endoscope zip-tied to the toolhead | LED-ring lit, VGA, moves with the nozzle |
| `lack_ir_night` | cantilever printer (A1 mini layout) in an IKEA Lack enclosure | security cam outside the acrylic | IR night mode: mono, noisy, timestamp OSD |

Each camera profile in `camera.py` sets the full device model:

- **Optics:** lens (rectilinear or equisolid fisheye), depth of field, bloom,
  barrel distortion, chromatic aberration, softness, vignetting
- **Sensor:** white-balance cast, signal-dependent shot and read noise,
  blotchy chroma noise, rolling-shutter banding, hot pixels, clipping
- **ISP and encoder:** saturation, contrast, oversharpening, burned-in OSD
  text, resolution, and JPEG quality
- **Exposure:** shutter time, which sets how much the moving parts motion-blur

## Clutter

Real printers don't sit in clean rooms, so every setup gets a seed-driven mess,
concentrated where its camera can see it:

- **Debris:** purge blobs, often two-tone from colour changes, piled by the
  chute and scattered on chamber floors and decks. Also curled filament
  offcuts, clumps of stringing, and a failed print balled up and tossed aside.
- **Build plate:** leftovers from earlier prints, kept clear of the current
  parts (strings, a small purge blob, a peeled skirt fragment).
- **Tools and supplies:** flush cutters, scraper, hex keys, glue stick, IPA
  spray bottle, painter's tape, sticky notes, paper, zip ties, a parts bin of
  old prints, spools (full, empty, lying down, hung on pegboard), a Raspberry
  Pi, and loose cables.
- **Per-setup extras:** the Voron has a purge bucket, nozzle brush and
  Nevermore filter. The Lack enclosure has a Pi on the panel, cable looms
  down the legs, and tape patches on the acrylic. The Prusa has a spool
  holder on the frame.
- **Dust:** dark, upward-facing surfaces collect patchy dust. The palette's
  `dust` amount sets how much per setup.

Loose filament is deliberately close in appearance to spaghetti but is not a
failure, so a detector can't learn "curvy filament means spaghetti". All of
it is in the `Environment` collection with pass index 1.

## build_scene.py failure: delamination then spaghetti (`--failure spaghetti`)

One part prints cleanly up to a random height (35-50% of the current Z).
Partway up that good section, the upper layers crack off and tilt 3-5.5 deg
about one long edge, with a small sideways shift. Every layer after that is
spaghetti. The script replays the layer's toolpath at each later nozzle
height, and the filament drifts, coils and drapes. It settles on whatever is
below it, using a heightfield that grows as strands pile up. The last strand
hangs up to the nozzle, and a blob of filament is wrapped around the hotend.
Any other parts on the plate keep printing normally. Heights, tilt, and strand
count are written to the scene's custom properties and to the `SCENE` log
line.

## Collections and masks

Objects sit in the `Environment`, `Printer`, `Print`, `Failure`, `CameraRig`,
and `Lights` collections. Pass indices follow CONTRACTS.md §5: 1 environment,
2 printer, 3 print, 4 spaghetti, 5 detached/delaminated, 6 shifted, 7 nozzle
blob, 8 occluder (hand, open door). `build_scene.py` stills still put their
nozzle blob in class 4.

## Adding a setup

1. Write a palette function and a build function in `setups.py`.
2. Reuse or add a printer builder in `printers.py` that returns a `Rig`.
   The toolhead nozzle tip goes at the carriage origin, and prints parent to
   `rig.bed`.
3. Pick or add a camera profile in `camera.py`.
4. Register the setup in `SETUPS`.

## Production runs

```bash
cd synthetic_data/pipeline
export UV_CACHE_DIR=/mnt/Primary/scythe-synth/corpus/raw/uv-cache
uv run synthpipe sample --run main1 --count 40000
uv run synthpipe run --run main1 --render-workers 3
```

Run with `--render-workers 3`. One Blender worker leaves the GPU about half
idle while it rebuilds geometry on the CPU between keyframes. Measured on the
RTX 5060 (2026-10-18):

| Render workers | Jobs/h | GPU utilisation |
|---|---|---|
| 1 | 484 | 47% |
| 2 | 724 | 65% |
| 3 | 844 | 85% |

Each worker uses about 1 GB of VRAM. Keep GPU training off the card while a
run is going, or cap it (see `training/README.md`).
