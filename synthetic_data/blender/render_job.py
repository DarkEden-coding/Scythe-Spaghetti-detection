"""Render one job spec to a directory, for testing outside the worker.

Usage:
    blender -b -P render_job.py -- --spec spec.json --out DIR [--png-preview]
    blender -b -P render_job.py -- --spec a.json --spec b.json --out DIR   # DIR/<job_id>/

With --png-preview, each keyframe also gets `k###.png` (plain sRGB of the
linear render, no ISP) and `k###_mask.png` (IndexOB as false colour, see
scenegen.imageio.CLASS_COLORS). --sheet DIR writes one contact sheet per job.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scenegen.sequence import render_sequence  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--spec", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--png-preview", action="store_true")
    p.add_argument("--closeup", action="store_true", help="debug: close camera on the part")
    p.add_argument("--sheet", default=None, help="write a contact sheet PNG per job here")
    return p.parse_args(argv)


def main():
    args = parse_args()
    for path in args.spec:
        with open(path) as f:
            spec = json.load(f)
        out = args.out if len(args.spec) == 1 else os.path.join(args.out, spec["job_id"])
        start = time.perf_counter()
        meta = render_sequence(spec, out, progress=lambda i: print(f"KEYFRAME {i}", flush=True),
                               png_preview=args.png_preview or bool(args.sheet),
                               closeup=args.closeup)
        print("JOB", spec["job_id"], f"{time.perf_counter() - start:.1f}s",
              json.dumps(meta["timing"]), flush=True)
        if args.sheet:
            from scenegen.sheet import contact_sheet
            os.makedirs(args.sheet, exist_ok=True)
            contact_sheet(out, spec, meta, os.path.join(args.sheet, f"{spec['job_id']}.png"))


if __name__ == "__main__":
    main()
