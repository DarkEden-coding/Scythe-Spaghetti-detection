"""
Blender script: build one 3D printer mid-print, seen through a cheap camera.

Usage:
    blender -b -P build_scene.py -- --setup prusa_desk --save scenes/prusa_desk.blend
    blender -b -P build_scene.py -- --setup voron_topcam --failure spaghetti --render f.jpg
    blender -b -P build_scene.py -- --setup lack_ir_night --seed 7 --samples 128 --render f.jpg

Setups (see scenegen/setups.py):
    enclosed_corexy   Bambu-style enclosed CoreXY, built-in chamber camera
    prusa_desk        Prusa MK4 on a desk, USB webcam on a mini tripod
    voron_topcam      Voron 2.4, fisheye Pi NoIR camera in the top corner
    ender_nozzlecam   Ender-3 with glass bed, endoscope on the toolhead
    lack_ir_night     Cantilever printer in a Lack enclosure, IR night camera

`--seed` varies the filament, the parts on the plate (count, shape, size,
placement, infill), print height, progress through the layer, the failure and
the sensor-noise pattern. Collections (Environment / Printer / Print / Failure
/ CameraRig / Lights) and pass indices are set for per-class masks.
"""

import argparse
import os
import random
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scenegen.camera import setup_isp, setup_render  # noqa: E402
from scenegen.clutter import plate_debris  # noqa: E402
from scenegen.materials import build_materials  # noqa: E402
from scenegen.printpart import build_print  # noqa: E402
from scenegen.setups import SETUPS, let_light_through_panels  # noqa: E402
from scenegen.util import LAYER, clear_scene, make_collections  # noqa: E402

FILAMENTS = {
    "orange PLA": (0.80, 0.20, 0.03),
    "galaxy grey PLA": (0.10, 0.10, 0.11),
    "white PLA": (0.78, 0.78, 0.74),
    "green PLA": (0.02, 0.45, 0.20),
    "red PETG": (0.60, 0.02, 0.03),
    "blue PLA": (0.03, 0.12, 0.55),
    "yellow PLA": (0.85, 0.60, 0.02),
    "purple PLA": (0.25, 0.04, 0.45),
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--setup", default="enclosed_corexy", choices=sorted(SETUPS))
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--failure", default="none", choices=("none", "spaghetti"),
                   help="spaghetti: part delaminates partway up, then spaghetti")
    p.add_argument("--filament", default=None, choices=sorted(FILAMENTS))
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--save", default=None, help="write the .blend here")
    p.add_argument("--render", default=None, help="render frame 1 to this path")
    return p.parse_args(argv)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    palette_fn, build_fn = SETUPS[args.setup]
    filament = args.filament or rng.choice(sorted(FILAMENTS))
    failed = args.failure == "spaghetti"

    clear_scene()
    colls = make_collections()
    palette = palette_fn(rng)
    palette["filament"] = FILAMENTS[filament]
    m = build_materials(palette)

    span = (0.018, 0.026) if failed else (0.008, 0.022)
    height = round(rng.uniform(*span) / LAYER) * LAYER
    setup = build_fn(m, colls, rng, height)
    let_light_through_panels(colls)
    nozzle, direction, info, parts = build_print(m, colls, rng, setup.rig, height, failed)
    plate_debris(m, colls, rng, setup.rig, parts, rng.randint(*setup.plate_debris))
    setup.rig.animate(nozzle, direction, setup.tool_speed)
    setup_render(setup.camera_profile, args.samples, setup.exposure)
    setup_isp(setup.camera_profile, args.seed)

    scene = bpy.context.scene
    meta = {"setup": args.setup, "seed": args.seed, "filament": filament,
            "failure": args.failure, "print_height_mm": round(height * 1000, 2), **info}
    for key, value in meta.items():
        scene[key] = value
    print("SCENE", meta)

    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save))
    if args.render:
        scene.render.filepath = os.path.abspath(args.render)
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
