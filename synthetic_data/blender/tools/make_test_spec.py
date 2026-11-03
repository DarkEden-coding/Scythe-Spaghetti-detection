"""Write a CONTRACTS.md §4 job spec for a given timeline + STL, for testing
render_job.py without the pipeline sampler. Needs numpy only.

    python make_test_spec.py --timeline T.npz --stl M.stl --out spec.json \
        [--t 900 | --frac 0.6] [--failure detached --mode tip --onset-frac 0.5] \
        [--event hand:880:905] [--setup prusa_desk] [--seed 3]

Keyframes are t-300, t-60, t-10 and t (those >= 0). With a failure and no
--t, t is placed `--after` seconds after onset.
"""

import argparse
import json

import numpy as np

PROFILE_SETUP = {"bambu_x1c": "enclosed_corexy", "prusa_mk4": "prusa_desk",
                 "voron24_300": "voron_topcam", "bambu_a1mini": "lack_ir_night"}
SETUP_CAMERA = {"enclosed_corexy": "chamber", "prusa_desk": "webcam",
                "voron_topcam": "pi_noir", "lack_ir_night": "ir_night"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeline", required=True)
    p.add_argument("--stl", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--setup", default=None)
    p.add_argument("--job-id", default="test")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--t", type=float, default=None)
    p.add_argument("--frac", type=float, default=0.6)
    p.add_argument("--failure", default=None,
                   choices=("spaghetti", "detached", "layer_shift", "nozzle_blob"))
    p.add_argument("--mode", default=None)
    p.add_argument("--onset-frac", type=float, default=0.45)
    p.add_argument("--after", type=float, default=240.0)
    p.add_argument("--event", action="append", default=[], help="type:t0:t1")
    p.add_argument("--keyframes", default=None, help="comma list, overrides history")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--copies", type=int, default=1)
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--filament", default="0.8,0.2,0.03")
    p.add_argument("--no-clutter", action="store_true")
    a = p.parse_args()

    with np.load(a.timeline, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"].item()))
    total = float(meta["total_time_s"])
    setup = a.setup or PROFILE_SETUP[meta["printer_profile"]]
    failure = None
    if a.failure:
        onset = total * a.onset_frac
        params = {"seed": a.seed}
        if a.mode:
            params["mode"] = a.mode
        failure = {"type": a.failure, "onset_s": onset, "params": params}
        t = a.t if a.t is not None else min(total, onset + a.after)
    else:
        t = a.t if a.t is not None else total * a.frac
    if a.keyframes:
        keys = sorted({float(x) for x in a.keyframes.split(",")})
    else:
        keys = sorted({round(t - d, 3) for d in (300, 60, 10, 0) if t - d >= 0})
    events = []
    for e in a.event:
        kind, t0, t1 = e.split(":")
        events.append({"type": kind, "t0": float(t0), "t1": float(t1),
                       "params": {"seed": a.seed}})
    camera = SETUP_CAMERA[setup]
    spec = {
        "job_id": a.job_id, "seed": a.seed, "generator_version": "test",
        "setup": setup, "printer_profile": meta["printer_profile"],
        "model": {"id": "test", "stl": a.stl, "license": "test"},
        "slice": {"layer_height": 0.2, "copies": a.copies, "scale": a.scale,
                  "gcode": meta.get("source_gcode", ""), "timeline": a.timeline},
        "filament": {"name": "test", "rgb": [float(x) for x in a.filament.split(",")]},
        "scene": {"variation_seed": a.seed * 7919, "light_scale": 1.0,
                  "camera_jitter": {"pos_m": 0.015, "rot_deg": 2.0},
                  "clutter": not a.no_clutter},
        "failure": failure, "keyframes": keys,
        "windows": [{"t": keys[-1], "history": keys[:-1][::-1]}],
        "events": events, "isp_variants": [{"profile": camera, "seed": 1}],
        "render": {"samples": a.samples,
                   "resolution": [640, 480 if camera == "pi_noir" else 360],
                   "motion_blur_shutter": 1.0 if camera == "ir_night" else 0.5},
    }
    with open(a.out, "w") as f:
        json.dump(spec, f, indent=1)
    print(a.out, keys)


if __name__ == "__main__":
    main()
