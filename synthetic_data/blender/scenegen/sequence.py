"""Render one job: a timeline-driven print sequence (CONTRACTS.md §4, §5).

`render_sequence(spec, out_dir, progress=None)` builds the scene once, then
for every keyframe time updates the print geometry, failure, toolhead pose
and render-side events, renders a multilayer EXR (Combined, IndexOB, Depth;
no compositor) and records per-class pixel counts from the IndexOB pass.
It leaves Blender empty for the next job in a persistent worker.
"""

import json
import math
import random
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Vector

from . import events as ev
from . import failures as fl
from .camera import PROFILES, setup_render
from .clutter import plate_debris
from .gcodeprint import PrintObjects, PrintPlan, normal_pieces
from .imageio import CLASS_COLORS, find_channel, linear_to_srgb8, read_exr, write_png
from .materials import build_materials
from .setups import SETUPS, let_light_through_panels
from .timeline import load_timeline
from .util import clear_scene, make_collections

SEQUENCE_SETUPS = ("enclosed_corexy", "prusa_desk", "voron_topcam", "lack_ir_night")
N_CLASSES = 9


class Footprint:
    """Box keep-out with the `sdf` interface clutter.plate_debris expects."""

    def __init__(self, box):
        self.box = box

    def sdf(self, x, y):
        x0, y0, x1, y1 = self.box
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        inside = min(x - x0, x1 - x, y - y0, y1 - y)
        return math.hypot(dx, dy) if (dx or dy) else -inside


def _jitter_camera(cam, rng, jitter):
    """Small seeded offset of the camera within its mount's envelope."""
    pos = float(jitter.get("pos_m", 0.0))
    rot = math.radians(float(jitter.get("rot_deg", 0.0)))
    cam.location += Vector([max(-1.0, min(1.0, rng.gauss(0, 0.5))) * pos for _ in range(3)])
    tweak = Euler((rng.uniform(-rot, rot), rng.uniform(-rot, rot), rng.uniform(-rot, rot) * 0.5))
    cam.rotation_euler = (cam.rotation_euler.to_quaternion() @ tweak.to_quaternion()).to_euler()


def camera_info(profile, resolution):
    cam = bpy.context.scene.camera.data
    if cam.type == "PANO":
        fov = math.degrees(cam.fisheye_fov)
        fisheye = True
    else:
        fov = math.degrees(2 * math.atan(cam.sensor_width / (2 * cam.lens)))
        fisheye = False
    return {"profile": profile, "resolution": list(resolution), "fov_deg": round(fov, 2),
            "fisheye": fisheye}


def configure_render(spec, setup):
    scene = bpy.context.scene
    r = spec.get("render", {})
    setup_render(setup.camera_profile, int(r.get("samples", 48)), setup.exposure)
    res = r.get("resolution") or PROFILES[setup.camera_profile]["resolution"]
    scene.render.resolution_x, scene.render.resolution_y = int(res[0]), int(res[1])
    scene.render.motion_blur_shutter = float(r.get("motion_blur_shutter",
                                                   PROFILES[setup.camera_profile]["shutter"]))
    # Fixed sampling seed. With the scene itself reproducible (see
    # util.noise_vector) the IndexOB/Depth passes are bit-identical between
    # runs; GPU OIDN leaves <0.01 linear differences in Combined (CPU OIDN
    # would remove them at ~4x the render time).
    scene.cycles.seed = int(spec.get("seed", 0)) % 2**31
    scene.cycles.use_animated_seed = False
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.denoising_use_gpu = True
    # Scenes have a handful of lights; the light tree costs more than it saves.
    scene.cycles.use_light_tree = False
    scene.cycles.max_bounces = 6
    scene.cycles.transparent_max_bounces = 8
    scene.render.use_persistent_data = True
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.compositing_node_group = None
    vl = scene.view_layers[0]
    vl.name = "ViewLayer"
    vl.use_pass_combined = True
    vl.use_pass_z = True
    vl.use_pass_object_index = True
    vl.pass_alpha_threshold = 0.999  # see-through panels never enter the mask
    vl.cycles.denoising_store_passes = False
    scene.view_settings.view_transform = "Standard"
    img = scene.render.image_settings
    img.media_type = "MULTI_LAYER_IMAGE"
    img.file_format = "OPEN_EXR_MULTILAYER"
    img.color_depth = "16"
    img.exr_codec = "ZIP"
    img.use_exr_interleave = True
    scene.render.film_transparent = False


def closeup_camera(ctx, angle=-35.0, dist=0.09):
    """Debug only: replace the camera with a close view of copy 0, parented
    to the bed so it follows a moving bed."""
    plan, rig = ctx["plan"], ctx["rig"]
    box = plan.footprint(0) if plan.copies else np.zeros(4)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    zc = plan.p1[plan.model_rows, 2].max() * 5e-4 if plan.model_rows.any() else 0.01
    cam = bpy.context.scene.camera
    cam.parent = rig.bed
    a = math.radians(angle)
    cam.location = (cx + dist * math.sin(a), cy - dist * math.cos(a), zc + dist * 0.75)
    cam.data.type = "PERSP"
    cam.data.lens = 18.0
    cam.data.sensor_width = 23.0
    cam.data.dof.use_dof = False
    from .util import aim
    aim(cam, (cx, cy, zc))


def model_stl(spec):
    """The STL exactly as the slicer loaded it (already scaled) when the
    slice cache kept it, else the corpus STL and the spec's scale."""
    sliced = Path(spec["slice"]["timeline"]).with_name("input.stl")
    if sliced.is_file():
        return str(sliced), 1.0
    return spec["model"]["stl"], float(spec["slice"].get("scale", 1.0))


def build(spec, log=print):
    """Build the scene for `spec`. Returns a context dict."""
    setup_name = spec["setup"]
    if setup_name not in SETUPS:
        raise ValueError(f"unknown setup {setup_name}")
    scene_spec = spec.get("scene", {})
    rng = random.Random(int(scene_spec.get("variation_seed", spec.get("seed", 0))))
    palette_fn, build_fn = SETUPS[setup_name]

    clear_scene()
    colls = make_collections()
    tl = load_timeline(spec["slice"]["timeline"])
    sl = spec.get("slice", {})
    stl, scale = model_stl(spec)
    plan = PrintPlan(tl, stl, scale, int(sl.get("copies", 1) or 1), log=log)

    palette = palette_fn(rng)
    palette["filament"] = tuple(spec.get("filament", {}).get("rgb", (0.8, 0.2, 0.03)))
    palette["layer"] = plan.layer_h * 1e-3
    m = build_materials(palette)

    # Rest pose only (every keyframe re-poses the rig); taken at t=0 so the
    # scene does not depend on which keyframes are rendered.
    height = float(plan.nozzle(0.0)[2])
    setup = build_fn(m, colls, rng, height, clutter=bool(scene_spec.get("clutter", True)))
    let_light_through_panels(colls)
    rig = setup.rig
    if scene_spec.get("clutter", True):
        keep = [Footprint(plan.footprint(c, 0.004)) for c in range(len(plan.copies))]
        plate_debris(m, colls, rng, rig, keep, rng.randint(*setup.plate_debris))
    _jitter_camera(bpy.context.scene.camera, rng, scene_spec.get("camera_jitter", {}))
    configure_render(spec, setup)

    pools = {"bed": PrintObjects(colls["Print"], rig.bed),
             "fail": PrintObjects(colls["Failure"], rig.bed),
             "tool": PrintObjects(colls["Failure"], rig.carriage)}
    ctx = {"spec": spec, "setup": setup, "setup_name": setup_name, "rig": rig, "m": m,
           "colls": colls, "plan": plan, "tl": tl, "pools": pools, "palette": palette,
           "light_scale": float(scene_spec.get("light_scale", 1.0))}
    ctx["failure"] = fl.build(spec.get("failure"), ctx)
    ctx["events"] = ev.Events(spec.get("events") or [], ctx)
    return ctx


def update(ctx, t):
    """Pose the scene for keyframe time t. Returns per-keyframe metadata."""
    plan, rig = ctx["plan"], ctx["rig"]
    events, failure = ctx["events"], ctx["failure"]
    te = events.print_time(t)  # part_removed restarts the print
    if te is None:
        pieces = []
        state = {"active": False, "severity": 0.0}
        pos = np.array(events.park_position())
        vel = np.zeros(3)
        telemetry = {"z_mm": 0.0, "layer": -1, "progress": 0.0, "elapsed_s": 0.0}
    else:
        plate_half = rig.plate_half
        if failure is None:
            pieces = normal_pieces(plan, te, ctx["m"], plate_half)
            state = {"active": False, "severity": 0.0}
            offset = np.zeros(3)
        else:
            pieces = failure.pieces(te)
            state = failure.state(te)
            offset = failure.nozzle_offset(te)
        pos = plan.nozzle(te) + offset
        vel = plan.velocity(te)
        telemetry = {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in ctx["tl"].state_at(te).items()}
    by_pool = {"bed": [], "fail": [], "tool": []}
    for piece in pieces:
        pool = piece[4] if len(piece) > 4 else ("bed" if piece[2] == 3 else "fail")
        by_pool[pool].append(piece[:4])
    stats = {}
    for name, pool in ctx["pools"].items():
        stats.update(pool.update(by_pool[name]))
    rig.animate_motion(tuple(pos), tuple(vel))
    active = events.apply(t, pos)
    return {"telemetry": telemetry, "nozzle_bed_mm": [round(float(v) * 1000, 3) for v in pos],
            "failure_state": state, "events_active": active, "faces": stats}


def class_pixels(exr_path):
    planes = read_exr(exr_path, want=("IndexOB", "Object Index"))
    if len(planes) != 1:
        raise KeyError(f"{exr_path}: expected one object index channel, got {sorted(planes)}")
    idx = next(iter(planes.values()))
    ids = np.rint(np.nan_to_num(idx.astype(np.float32))).astype(np.int64)
    ids = np.clip(ids, 0, N_CLASSES - 1)
    counts = np.bincount(ids.ravel(), minlength=N_CLASSES)
    return {str(i): int(c) for i, c in enumerate(counts) if i > 0}, ids


def mark_visible(state, pixels):
    """`visible` is what the render shows: the failure is active and its
    class has pixels in the IndexOB pass. The generator's own expectation
    is kept as `expected_visible`."""
    cls = fl.FAILURE_CLASS.get(state.get("type"))
    if cls is None:
        return
    n = int(pixels.get(str(cls), 0))
    state["expected_visible"] = bool(state.get("visible", False))
    state["visible_px"] = n
    state["visible"] = bool(state.get("active")) and n > 0


def write_previews(exr_path, png_path, ids):
    planes = read_exr(exr_path, want=("Combined",))
    rgb = np.stack([find_channel(planes, "Combined", c) for c in "RGB"], axis=-1)
    write_png(png_path, linear_to_srgb8(rgb))
    write_png(str(png_path).replace(".png", "_mask.png"), CLASS_COLORS[ids])


def render_sequence(spec, out_dir, progress=None, png_preview=False, log=print,
                    closeup=False):
    """Render every keyframe of `spec` into `out_dir` (CONTRACTS.md §5)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    try:
        ctx = build(spec, log=log)
        if closeup:
            closeup_camera(ctx)
        build_s = time.perf_counter() - t_start
        scene = bpy.context.scene
        frames = []
        timing = []
        for idx, t in enumerate(spec["keyframes"]):
            t0 = time.perf_counter()
            info = update(ctx, float(t))
            t1 = time.perf_counter()
            scene.frame_set(1)
            path = out_dir / f"k{idx:03d}.exr"
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            t2 = time.perf_counter()
            pixels, ids = class_pixels(path)
            mark_visible(info["failure_state"], pixels)
            if png_preview:
                write_previews(path, out_dir / f"k{idx:03d}.png", ids)
            t3 = time.perf_counter()
            frames.append({"idx": idx, "t": float(t), "telemetry": info["telemetry"],
                           "nozzle_bed_mm": info["nozzle_bed_mm"],
                           "failure_state": info["failure_state"],
                           "class_pixels": pixels, "events_active": info["events_active"]})
            timing.append({"update_s": round(t1 - t0, 3), "render_s": round(t2 - t1, 3),
                           "readback_s": round(t3 - t2, 3), "faces": info["faces"]})
            if progress is not None:
                progress(idx)
        total = time.perf_counter() - t_start
        align = ctx["plan"].alignment
        meta = {"job_id": spec.get("job_id"), "blender_version": bpy.app.version_string,
                "alignment_ok": align["ok"], "alignment": align,
                "render_seconds": round(total, 2),
                "camera": camera_info(ctx["setup"].camera_profile,
                                      (scene.render.resolution_x, scene.render.resolution_y)),
                "keyframes": frames,
                "timing": {"build_s": round(build_s, 3), "keyframes": timing}}
        with open(out_dir / "render_meta.json", "w") as f:
            json.dump(meta, f, indent=1)
        return meta
    finally:
        clear_scene()
