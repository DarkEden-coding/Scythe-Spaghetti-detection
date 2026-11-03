# G-code timeline regression fixtures and validation

Generated locally with OrcaSlicer 2.4.2 on 2026-10-18, independently of synthpipe.slicing.
All executable instructions and useful header/footer metadata are preserved. Only long (>500 characters) serialized configuration templates were omitted to keep the entire fixture directory under 2 MB.

The generated compound STL contains a 12×12×8 mm cube, a radius-5 mm / height-12 mm cylinder (64 sides), and a 4×5×10 mm post with a 14×8×3 mm overhanging cap. The profiles use their bundled default process/filament, normal supports enabled, and a 2 mm outer brim. Resolved JSON presets, the source STL, generation scripts, untrimmed G-code, and reports are in `/mnt/Primary/scythe-synth/tmp_w2`.

Commands (Orca subprocesses must run with cwd set to their scratch output directory):

```bash
cd "/mnt/Fast Files/Ceph-Mirror/Python-Files/Projects/Scythe-Spaghetti-detection/synthetic_data/pipeline"
uv run python /mnt/Primary/scythe-synth/tmp_w2/slice_fixtures.py
# Each profile invokes:
flatpak run --branch=stable --filesystem=/mnt/Primary com.orcaslicer.OrcaSlicer --debug 1 --load-settings "$SCRATCH/machine.json;$SCRATCH/process.json" --load-filaments "$SCRATCH/filament.json" --arrange 1 --orient 0 --slice 0 --outputdir "$SCRATCH" /mnt/Primary/scythe-synth/tmp_w2/shapes.stl
uv run --extra dev pytest tests/test_gcode.py -q
uvx ruff check synthpipe/gcode.py tests/test_gcode.py ../blender/scenegen/timeline.py
uv run synthpipe timeline /mnt/Primary/scythe-synth/tmp_w2/bambu_x1c/plate_1.gcode -o /mnt/Primary/scythe-synth/tmp_w2/bambu_x1c/timeline.npz
uv run synthpipe timeline --all --workers 4
uv run python /mnt/Primary/scythe-synth/tmp_w2/benchmark.py
```

Validation: 45 pytest cases pass, including fixture-based cross-reader queries, schema/dtypes, modes/resets, retractions, arcs/helixes and chord error, feature codes, layer transitions/hops, acceleration and junction/cruise limits, and resumable/parallel CLI output. Pipeline and Blender Timeline class ASTs are identical. The reader also loaded and queried an archive inside Blender 5.2.1 / Python 3.13.13.

Timing results (no empirical scaling to match the header):

- `bambu_a1mini`: 17,613 rows / 99 layers; modeled 600.09 s; header 639 s; raw ratio 0.9391. Print-only comparison 639 s, ratio 0.9391.
- `bambu_x1c`: 16,373 rows / 99 layers; modeled 449.33 s; header 444 s; raw ratio 1.0120. Print-only comparison 444 s, ratio 1.0120.
- `prusa_mk4`: 15,550 rows / 81 layers; modeled 752.82 s; header 2299 s; raw ratio 0.3275. Print-only comparison 720 s, ratio 1.0456.
- `voron24_300`: 15,715 rows / 89 layers; modeled 527.38 s; header 534 s; raw ratio 0.9876. Print-only comparison 480 s, ratio 1.0987.

MK4 header time includes startup heating before print proper. The separate print-only comparison uses the last emitted M73 R value at the first actual print extrusion, rounded to minutes. The raw header value and ratio are always retained; there is no calibration against either estimate. Four tiny integration cubes in the first production-cache audit were outside ±15% against this rounded estimate (ratio 1.247). Corpus-size models in that initial small audit were within the target; the expanded audit below includes additional outliers.

Bed polygons are read from the G-code configuration, not from printer-name constants. Verified size/centre: X1C 256×256 / (128,128), A1 mini 180×180 / (90,90), MK4 250×210 / (125,105), Voron 300×300 / (150,150), all mm. Blender mapping is `(x-cx, y-cy, z)/1000`, giving metres with the plate surface centre as origin. Missing bed metadata stays null and logs a warning.

Startup convention: unmarked startup/custom and explicit purge/prime/wipe extrusion is kind 9 and retained at negative times. t=0 is the first print-proper moving extrusion (including brim/support). Other earlier startup motion and all dwell/wait time are omitted. Travels after print start are retained. Layers include transition travels starting at the layer-change comment; layer Z comes from declared print Z, not hops. Positive stationary E rows do not produce partial geometry.

Stress benchmark: independently sliced 150×150×30 mm Voron cube, 0.12 mm layers / 40% gyroid. 309,479,764 bytes, 10,927,779 lines, 10,914,011 rows, 249 layers; 44.23 s parse, 247,085 lines/s (7.00 MB/s), 3789 MiB peak RSS. Modeled 122097.32 s vs header 127941 s (ratio 0.9543). The 1M-lines/s throughput target is not met; this is a packed-array streaming Python implementation, and peak memory still scales with the resulting timeline.

Production cache: completed all **602 timelines** (600 corpus slices plus two MK4 integration slices), zero missing archives and no parser errors. Every cached timeline profile and plate centre agrees with its adjacent slice.json. The CLI ran resumably while the concurrent slicer grew the pool; scoped A1 mini and Voron catch-up passes avoided duplicating files already in flight. The final all-cache resume is a no-op. Reports and command logs are in the scratch directory; `cache_validation_latest.json` includes every timing outlier.

Expanded cache print-only ratios (modeled/slicer):
- `bambu_a1mini`: 150 archives; median 0.9722; range 0.7501–1.4224; 140/150 within ±15%.
- `bambu_x1c`: 150 archives; median 0.9742; range 0.6591–1.6346; 128/150 within ±15%.
- `prusa_mk4`: 152 archives; median 1.0257; range 0.8090–1.4704; 137/152 within ±15%.
- `voron24_300`: 150 archives; median 1.0146; range 0.9307–1.7128; 133/150 within ±15%.

Overall, 538/600 corpus slices (89.7%) meet ±15%. The worst expanded-cache ratios range from 0.659 to 1.713: this timing model is approximate, and the per-profile medians do not imply a per-file guarantee. Keep these outliers visible when selecting a timing calibration subset.

The largest shared-cache G-code (Voron thingi10k_90435, 736.9 MB) also parsed successfully: 128.17 s parsing plus 27.56 s archive writing, ratio 0.9666 against its header estimate. The separate reproducible 309 MB stress benchmark above records isolated parse speed and peak memory.

Known limits: firmware macro bodies (Voron PRINT_START/PRINT_END) are unavailable; Bambu firmware conditionals are not evaluated, so their written move branches are replayed. Waits/dwells are excluded. Junction/cruise timing uses local approximations rather than full firmware lookahead; classic per-axis jerk is not modeled. No contract edits were made. No commits or git state changes were performed.
