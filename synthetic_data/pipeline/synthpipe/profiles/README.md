# OrcaSlicer system presets

The twelve standalone JSON files are flattened copies of the stable 2.4.2
Flatpak's `/app/share/OrcaSlicer/profiles` presets. `sources.json` records each
preset's original name and full inheritance chain. Source:
https://github.com/OrcaSlicer/OrcaSlicer/tree/v2.4.2/resources/profiles
Upstream OrcaSlicer distributes these presets under AGPL-3.0.

Re-export with:

```sh
uv run python synthpipe/profiles/export_profiles.py
```

The exporter handles Flatpak hard-linked common presets and vendor-scoped
inheritance. It refuses a different slicer version or ambiguous inheritance.
The Prusa MK4 0.4 system preset includes input-shaper firmware setup. All
filaments are bundled PLA presets; X1C uses Bambu PLA Basic, A1 mini uses
Generic PLA, Prusa uses Prusa Generic PLA, and Voron uses Generic PLA @System.

`slice_model` writes the effective presets alongside each slice. It overrides
only randomized print settings plus comments, disabled arc fitting, disabled
prime tower, and the Textured PEI plate type. Original startup, shutdown and
firmware flavor settings remain in effect. Bambu's native startup wiping
arcs remain, even with print-toolpath arc fitting disabled. Voron uses the
firmware's PRINT_START / PRINT_END macros, whose internal motions cannot be
reconstructed from exported G-code alone.

Orca 2.4.2's raw-STL CLI `--repetitions` is unsuitable; cloning uses
`--clone-objects`. Its CLI `--scale` crashes for enlargement on this install,
so scaling is applied to a temporary STL before Orca loads it. Originals
remain unscaled in the corpus.
