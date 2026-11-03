"""Named setups: printer + surroundings + lights + camera mount.

Each setup has a palette (fed to `build_materials`) and a build function that
returns a `SetupResult`. World layout: the printer stands on z = 0 and faces
-Y; its nozzle works around x = y = 0 (bed slingers) or the plate centre.
"""

import math
from dataclasses import dataclass

from . import clutter as cl
from .camera import build_camera
from .materials import emission, principled
from .printers import (BAMBU, VORON, Rig, build_bambu, build_cantilever, build_ender3,
                       build_prusa_mk4, build_voron)
from .util import (PASS_ENVIRONMENT, area_light, box, catmull, cylinder, fan,
                   point_light, ring, set_world, tube)


@dataclass
class SetupResult:
    rig: Rig
    camera_profile: str
    exposure: float
    tool_speed: float  # m/s along the current line, drives motion blur
    plate_debris: tuple = (1, 3)  # range of leftover bits on the build plate


# ---------------------------------------------------------------------------
# Shared props
# ---------------------------------------------------------------------------

TRANSPARENT = {"Glass", "SmokedAcrylic", "ClearAcrylic"}


def let_light_through_panels(colls):
    """With refractive caustics off, Cycles treats glass as opaque to lights,
    so a lamp outside an enclosure would light nothing inside. Panels still
    reflect and refract; they just stop casting shadows."""
    for obj in colls["Environment"].all_objects:
        if obj.type == "MESH" and any(mat and mat.name in TRANSPARENT
                                      for mat in obj.data.materials):
            obj.visible_shadow = False

def spool(m, c, center, axis="X"):
    """Spare filament spool lying around: flanges, wound filament, hub."""
    normal = {"X": (1, 0, 0), "Y": (0, 1, 0)}[axis]
    for s in (-1, 1):
        off = tuple(center[i] + s * 0.032 * normal[i] for i in range(3))
        cylinder(f"SpoolFlange{s:+d}", off, 0.1, 0.004, m["spool_flange"], c, axis=axis,
                 verts=64, pass_index=PASS_ENVIRONMENT)
    cylinder("SpoolWinding", center, 0.085, 0.06, m["spool"], c, axis=axis, verts=64,
             pass_index=PASS_ENVIRONMENT)
    cylinder("SpoolHub", center, 0.028, 0.066, m["void"], c, axis=axis, verts=32,
             pass_index=PASS_ENVIRONMENT)


# ---------------------------------------------------------------------------
# 1. Enclosed CoreXY, built-in chamber camera
# ---------------------------------------------------------------------------

def palette_enclosed_corexy(rng):
    return {"frame": "black", "accent": (0.05, 0.05, 0.055), "plate": "textured_gold",
            "dust": 0.35}


def build_enclosed_corexy(m, colls, rng, height, clutter=True):
    env, lights = colls["Environment"], colls["Lights"]
    x, h = 0.19, 0.43
    yf, yb = BAMBU["y_front"], BAMBU["y_back"]
    t, depth, yc = 0.008, yb - yf, (yf + yb) / 2

    def panel(name, center, size, mat):
        box(name, center, size, mat, env, PASS_ENVIRONMENT)

    panel("Floor", (0, yc, -t / 2), (2 * x + 2 * t, depth, t), m["env_dark"])
    panel("WallLeft", (-x - t / 2, yc, h / 2), (t, depth, h), m["env_dark"])
    panel("WallRight", (x + t / 2, yc, h / 2), (t, depth, h), m["env_dark"])
    panel("WallBack", (0, yb + t / 2, h / 2), (2 * x + 2 * t, t, h), m["sheet"])
    rim = 0.025
    panel("TopRimFront", (0, yf + rim / 2, h + t / 2), (2 * x, rim, t), m["sheet"])
    panel("TopRimBack", (0, yb - rim / 2, h + t / 2), (2 * x, rim, t), m["sheet"])
    panel("TopRimLeft", (-x + rim / 2, yc, h + t / 2), (rim, depth, t), m["sheet"])
    panel("TopRimRight", (x - rim / 2, yc, h + t / 2), (rim, depth, t), m["sheet"])
    panel("GlassLid", (0, yc, h + t + 0.002), (2 * x, depth, 0.004), m["glass"])
    panel("DoorGlass", (0, yf - 0.006, h / 2), (2 * x, 0.004, h), m["glass"])
    panel("DoorFrameL", (-x + 0.006, yf - 0.002, h / 2), (0.012, 0.006, h), m["sheet"])
    panel("DoorFrameR", (x - 0.006, yf - 0.002, h / 2), (0.012, 0.006, h), m["sheet"])
    fan("Exhaust", (0.09, yb - 0.002, 0.22), 0.046, "Y", m, env,
        pass_index=PASS_ENVIRONMENT)
    fan("AuxFan", (-x + 0.002, 0.02, 0.2), 0.058, "-X", m, env, pass_index=PASS_ENVIRONMENT)
    box("PurgeChute", (-0.15, yb - 0.025, 0.06), (0.05, 0.04, 0.12), m["env_dark"], env,
        PASS_ENVIRONMENT, bevel=0.004)

    rig = build_bambu(m, colls, height)

    if clutter:
        # Purge waste that missed the chute piles up beside it; the rest of the
        # floor collects strands, offcuts and dust.
        for _ in range(rng.randint(6, 14)):
            cl.purge_blob(m, env, rng, (-0.15 + rng.gauss(0, 0.025), yb - 0.075 + rng.gauss(0, 0.02),
                                        0.0))
        cl.debris(m, env, rng, (-x + 0.01, x - 0.01, yf + 0.01, yb - 0.05), 0.0,
                  purge=rng.randint(6, 12), offcuts=rng.randint(3, 6), hair=rng.randint(2, 5))
        cl.flush_cutters(m, env, rng, (rng.uniform(-0.15, -0.05), yf + 0.04, 0.0),
                         rng.uniform(-0.5, 0.5))
        # Painter's tape holding a cable to the left wall, and a sticker.
        box("TapeStrip", (-x + 0.0005, 0.08, 0.33), (0.001, 0.05, 0.018),
            principled("Tape", (0.05, 0.25, 0.65), 0.7), env, PASS_ENVIRONMENT,
            rotation=(rng.uniform(-0.2, 0.2), 0, 0))
        cl.loose_cable(env, rng, (-x + 0.004, -0.1), (-x + 0.004, 0.17), 0.33, radius=0.0015,
                       slack=0.0)

    # Chamber LED bar next to the camera (~6000 K), warm room light through
    # the glass, dim ambient. (1.6 W: at 1.3 W sequence frames came out
    # darker than the preview once the ISP added its noise.)
    area_light("ChamberLED", lights, (-0.08, yf + 0.03, h - 0.006), (0.0, 0.05, 0.0),
               1.6, (0.90, 0.95, 1.0), 0.11, 0.012)
    box("LEDDiffuser", (-0.08, yf + 0.03, h - 0.003), (0.115, 0.016, 0.004), m["sheet"],
        env, PASS_ENVIRONMENT)
    area_light("RoomLamp", lights, (0.4, -0.9, 0.9), (0, 0, 0.2), 8.0, (1.0, 0.82, 0.62),
               0.8, shape="DISK")
    set_world((0.32, 0.29, 0.26), 0.2)

    build_camera("ChamberCam", "chamber", colls["CameraRig"],
                 (-0.152, yf + 0.014, h - 0.022), (-0.025, 0.07, 0.19))
    housing = box("CamHousing", (-0.176, yf + 0.008, h - 0.024), (0.026, 0.014, 0.022),
                  m["body"], colls["CameraRig"], PASS_ENVIRONMENT, bevel=0.003)
    housing.rotation_euler = (0, 0, math.radians(45))
    return SetupResult(rig, "chamber", 0.0, 0.15)


# ---------------------------------------------------------------------------
# 2. Prusa MK4 on a desk, USB webcam on a mini tripod
# ---------------------------------------------------------------------------

def palette_prusa_desk(rng):
    return {"frame": "black", "accent": (0.8, 0.16, 0.012), "plate": "satin_black",
            "wall": (0.42, 0.45, 0.4), "spool": rng.choice(
                [(0.02, 0.25, 0.6), (0.7, 0.7, 0.68), (0.03, 0.03, 0.03), (0.6, 0.05, 0.3)]),
            "mug": (0.75, 0.75, 0.72), "dust": 0.15}


def build_prusa_desk(m, colls, rng, height, clutter=True):
    env, lights = colls["Environment"], colls["Lights"]
    box("Desk", (0, 0.05, -0.015), (1.6, 0.8, 0.03), m["wood"], env, PASS_ENVIRONMENT)
    box("Wall", (0, 0.46, 0.7), (3.0, 0.02, 1.5), m["paint"], env, PASS_ENVIRONMENT)
    box("SideWall", (-0.85, 0.0, 0.7), (0.02, 2.0, 1.5), m["paint"], env, PASS_ENVIRONMENT)
    box("Floor", (0, 0, -0.75), (4, 4, 0.02), m["concrete"], env, PASS_ENVIRONMENT)
    # Window on the back wall to the right: bright, slightly blue daylight.
    sky = emission("WindowSky", (0.75, 0.85, 1.0), 6.0)
    box("Window", (0.62, 0.448, 0.62), (0.55, 0.004, 0.6), sky, env, PASS_ENVIRONMENT)
    for i, (cx, cz, sx, sz) in enumerate(((0.62, 0.32, 0.6, 0.03), (0.62, 0.92, 0.6, 0.03),
                                          (0.335, 0.62, 0.03, 0.6), (0.905, 0.62, 0.03, 0.6),
                                          (0.62, 0.62, 0.02, 0.6))):
        box(f"WindowFrame{i}", (cx, 0.444, cz), (sx, 0.012, sz), m["lack"], env,
            PASS_ENVIRONMENT)
    area_light("Daylight", lights, (0.62, 0.43, 0.62), (0.2, -0.3, 0.0), 30.0,
               (0.85, 0.92, 1.0), 0.55, 0.6)
    area_light("CeilingLamp", lights, (-0.3, -0.4, 1.4), (0, 0, 0), 60.0, (1.0, 0.78, 0.55),
               0.4, shape="DISK")
    set_world((0.35, 0.33, 0.3), 0.25)

    spool(m, env, (-0.4, 0.2, 0.1))
    box("ShippingBox", (0.45, 0.2, 0.08), (0.22, 0.18, 0.16), m["cardboard"], env,
        PASS_ENVIRONMENT, bevel=0.002)
    cylinder("Mug", (0.36, -0.22, 0.045), 0.04, 0.09, m["ceramic"], env, verts=40,
             pass_index=PASS_ENVIRONMENT)
    ring("MugHandle", (0.4, -0.22, 0.05), 0.022, 0.006, m["ceramic"], env, axis="Y",
         pass_index=PASS_ENVIRONMENT)

    rig = build_prusa_mk4(m, colls, height)

    if clutter:
        printer = (-0.25, 0.26, -0.27, 0.22)
        keep_out = [printer, (0.17, 0.37, -0.56, -0.34), (-0.52, -0.28, 0.08, 0.32),
                    (0.32, 0.58, 0.09, 0.31), (0.3, 0.44, -0.28, -0.16)]
        # The webcam sees the desk on either side of the printer, so the mess
        # concentrates there.
        for region in ((-0.62, -0.26, -0.3, 0.4), (0.27, 0.62, -0.3, 0.4)):
            cl.desk_clutter(m, env, rng, region, keep_out, count=rng.randint(5, 8))
        for x, y in cl.scatter(rng, rng.randint(1, 3), (-0.75, 0.75, -0.1, 0.38), keep_out,
                               spacing=0.22):
            cl.spool_lying(m, env, rng, (x, y, 0.0), rng.uniform(0, 6.3),
                           empty_spool=rng.random() < 0.3)
            keep_out.append((x - 0.11, x + 0.11, y - 0.11, y + 0.11))
        cl.paper(env, rng, (rng.uniform(-0.5, -0.3), -0.34, 0.0), rng.uniform(-0.5, 0.5))
        # Spool holder on top of the frame, as shipped.
        for s in (-1, 1):
            box(f"SpoolArm{s:+d}", (s * 0.04, 0.078, 0.5), (0.012, 0.03, 0.08), m["accent"], env,
                PASS_ENVIRONMENT, bevel=0.003)
        cylinder("SpoolAxle", (0, 0.078, 0.54), 0.012, 0.11, m["accent"], env, axis="X",
                 pass_index=PASS_ENVIRONMENT)
        spool(m, env, (0, 0.078, 0.54))
        cl.raspberry_pi(m, env, rng, (0.3, 0.3, 0.0), rng.uniform(0, 6.3))
        cl.loose_cable(env, rng, (0.3, 0.3), (0.1, 0.24), 0.0)      # Pi -> printer USB
        cl.loose_cable(env, rng, (0.0, 0.25), (0.25, 0.42), 0.0)    # PSU cable to the wall
        cl.loose_cable(env, rng, (0.32, -0.4), (0.7, 0.3), 0.0, radius=0.0018)  # webcam USB

    cam_pos = (0.27, -0.44, 0.2)
    cam = build_camera("Webcam", "webcam", colls["CameraRig"], cam_pos, (0.0, 0.0, 0.11))
    box("WebcamBody", (0, 0, 0.02), (0.09, 0.026, 0.03), m["body"], colls["CameraRig"],
        PASS_ENVIRONMENT, bevel=0.008, parent=cam)
    for i in range(3):
        a = 2 * math.pi * i / 3
        foot = (cam_pos[0] + 0.07 * math.cos(a), cam_pos[1] + 0.07 * math.sin(a), 0.0)
        tube(f"TripodLeg{i}", [[(cam_pos[0], cam_pos[1], cam_pos[2] - 0.03), foot]], 0.004,
             m["body"], colls["CameraRig"], pass_index=PASS_ENVIRONMENT)
    return SetupResult(rig, "webcam", 0.0, 0.12)


# ---------------------------------------------------------------------------
# 3. Voron 2.4 with a fisheye Pi NoIR camera in the top front corner
# ---------------------------------------------------------------------------

def palette_voron(rng):
    return {"frame": "black", "accent": (0.55, 0.012, 0.01), "plate": "textured_black",
            "wall": (0.3, 0.3, 0.32), "dust": 0.25}


def build_voron_topcam(m, colls, rng, height, clutter=True):
    env, lights = colls["Environment"], colls["Lights"]
    w, top = VORON["half_w"], VORON["top"]
    box("Floor", (0, 0, -0.01), (4, 4, 0.02), m["concrete"], env, PASS_ENVIRONMENT)
    box("Wall", (0, 1.0, 1.0), (4, 0.02, 2), m["paint"], env, PASS_ENVIRONMENT)
    rig = build_voron(m, colls, height)
    deck = VORON["deck"] + 0.0035
    bed_z = VORON["bed_z"]

    # Purge bucket and nozzle brush at the back edge of the bed, overflowing.
    bucket = cl.root("PurgeBucket", env, (0.17, 0.185, bed_z - 0.02), 0.0)
    for name, ctr, size in (("BucketFloor", (0, 0, 0.001), (0.05, 0.03, 0.002)),
                            ("BucketFront", (0, -0.014, 0.012), (0.05, 0.002, 0.024)),
                            ("BucketBack", (0, 0.014, 0.012), (0.05, 0.002, 0.024)),
                            ("BucketL", (-0.024, 0, 0.012), (0.002, 0.03, 0.024)),
                            ("BucketR", (0.024, 0, 0.012), (0.002, 0.03, 0.024))):
        box(name, ctr, size, m["accent"], env, PASS_ENVIRONMENT, parent=bucket)
    if clutter:
        for _ in range(rng.randint(4, 8)):
            cl.purge_blob(m, env, rng, (rng.uniform(-0.018, 0.018), rng.uniform(-0.01, 0.01),
                                        rng.uniform(0.002, 0.02)), parent=bucket)
        box("NozzleBrush", (0.11, 0.185, bed_z + 0.004), (0.04, 0.008, 0.012),
            principled("Brush", (0.6, 0.6, 0.6), 0.6, metallic=0.6), env, PASS_ENVIRONMENT)
        # Nevermore filter on the deck, cables loose on the deck, debris everywhere.
        cylinder("Nevermore", (-0.17, 0.17, deck + 0.04), 0.035, 0.08, m["accent"], env,
                 pass_index=PASS_ENVIRONMENT)
        fan("NevermoreFan", (-0.17, 0.17, deck + 0.081), 0.025, "Z", m, env,
            pass_index=PASS_ENVIRONMENT)
        for _ in range(3):
            cl.loose_cable(env, rng, (rng.uniform(-0.2, 0.2), -0.2), (rng.uniform(-0.2, 0.2), 0.2),
                           deck, radius=0.0018)
        cl.debris(m, env, rng, (-0.22, 0.22, -0.22, 0.22), deck, purge=rng.randint(4, 9),
                  offcuts=rng.randint(2, 5), hair=rng.randint(2, 4))
        cl.scraper(m, env, rng, (rng.uniform(-0.12, 0.05), -0.19, deck), rng.uniform(-0.6, 0.6))
        for x, y in cl.scatter(rng, rng.randint(2, 4), (-0.21, 0.21, -0.21, -0.12), spacing=0.06):
            cl.old_print(m, env, rng, (x, y, deck), rng.uniform(0, 6.3))
        cl.zip_ties(env, rng, (rng.uniform(-0.15, 0.15), 0.19, deck))
        cl.hex_keys(m, env, rng, (0.12, -0.17, deck), rng.uniform(0, 6.3))

    # Cool-white LED strip under the top front extrusion.
    strip = emission("LEDStrip", (0.92, 0.96, 1.0), 12.0)
    box("LEDStrip", (0, -w + 0.012, top - 0.024), (2 * w - 0.04, 0.008, 0.002), strip, env,
        PASS_ENVIRONMENT)
    area_light("LEDStripLight", lights, (0, -w + 0.015, top - 0.03), (0, 0.05, 0.0), 9.0,
               (0.92, 0.96, 1.0), 2 * w - 0.04, 0.01)
    area_light("RoomLight", lights, (-0.8, -1.2, 1.8), (0, 0, 0.3), 120.0,
               (1.0, 0.9, 0.78), 1.0, shape="DISK")
    set_world((0.2, 0.2, 0.22), 0.15)

    cam = build_camera("PiCam", "pi_noir", colls["CameraRig"],
                       (0.19, -0.19, top - 0.07), (-0.01, 0.02, VORON["bed_z"]))
    box("PiCamBoard", (0, 0, 0.006), (0.025, 0.024, 0.002), m["pcb"], colls["CameraRig"],
        PASS_ENVIRONMENT, parent=cam)
    box("PiCamMount", (0, 0, 0.012), (0.03, 0.03, 0.01), m["accent"], colls["CameraRig"],
        PASS_ENVIRONMENT, bevel=0.002, parent=cam)
    return SetupResult(rig, "pi_noir", 0.0, 0.2, (2, 4))


# ---------------------------------------------------------------------------
# 4. Ender-3 in a garage, USB endoscope zip-tied to the toolhead
# ---------------------------------------------------------------------------

def palette_ender(rng):
    return {"frame": "silver", "accent": (0.02, 0.09, 0.45), "plate": "glass",
            "body": (0.02, 0.02, 0.022), "dust": 0.2}


def build_ender_nozzlecam(m, colls, rng, height, clutter=True):
    env, lights = colls["Environment"], colls["Lights"]
    box("Workbench", (0, 0.05, -0.02), (1.4, 0.75, 0.04), m["wood"], env, PASS_ENVIRONMENT)
    box("Pegboard", (0, 0.42, 0.55), (1.4, 0.006, 1.0), m["pegboard"], env, PASS_ENVIRONMENT)
    box("Floor", (0, 0, -0.9), (4, 4, 0.02), m["concrete"], env, PASS_ENVIRONMENT)
    spool(m, env, (0.42, 0.25, 0.1), axis="X")
    tubelight = emission("FluorescentTube", (0.9, 1.0, 0.95), 15.0)
    cylinder("FluorescentTube", (0.1, -0.1, 1.1), 0.013, 1.2, tubelight, lights, axis="X",
             pass_index=PASS_ENVIRONMENT)
    area_light("TubeLight", lights, (0.1, -0.1, 1.08), (0.1, -0.1, 0.0), 80.0,
               (0.9, 1.0, 0.95), 1.2, 0.04)
    set_world((0.25, 0.27, 0.3), 0.2)

    rig = build_ender3(m, colls, height)

    if clutter:
        keep_out = [(-0.32, 0.33, -0.27, 0.26), (0.32, 0.52, 0.15, 0.35)]
        cl.desk_clutter(m, env, rng, (-0.68, 0.68, -0.34, 0.38), keep_out,
                        count=rng.randint(8, 12))
        for x, y in cl.scatter(rng, 2, (-0.68, 0.68, 0.0, 0.36), keep_out, spacing=0.22):
            cl.spool_lying(m, env, rng, (x, y, 0.0), rng.uniform(0, 6.3))
        # Spools hung on the pegboard.
        for i, x in enumerate((-0.45, -0.22, 0.25)):
            cylinder(f"PegHook{i}", (x, 0.38, 0.55), 0.003, 0.08, m["steel"], env, axis="Y",
                     pass_index=PASS_ENVIRONMENT)
            hung = cl.spool_lying(m, env, rng, (x, 0.34, 0.55), 0.0)
            hung.rotation_euler = (math.pi / 2, 0, 0)
            hung.location = (x, 0.41, 0.55)
        cl.loose_cable(env, rng, (0.27, 0.06), (0.6, 0.38), 0.0)

    cam = build_camera("Endoscope", "endoscope", colls["CameraRig"],
                       (-0.06, -0.055, 0.03), (0.0, 0.012, -0.006), parent=rig.carriage)
    cylinder("EndoscopeBody", (0, 0, 0.022), 0.0045, 0.04, m["steel"], colls["CameraRig"],
             parent=cam, pass_index=PASS_ENVIRONMENT)
    box("ZipTieMount", (0.01, 0, 0.03), (0.016, 0.012, 0.012), m["accent"],
        colls["CameraRig"], PASS_ENVIRONMENT, bevel=0.002, parent=cam)
    # LED ring around the lens.
    ring("LEDRing", (0, 0, 0.0005), 0.0035, 0.0006,
         emission("EndoscopeLED", (1.0, 0.97, 0.9), 40.0), colls["CameraRig"],
         pass_index=PASS_ENVIRONMENT, parent=cam)
    point_light("EndoscopeLight", lights, (0, 0, 0.0), 0.3, (1.0, 0.96, 0.88),
                radius=0.003, parent=cam, spot=math.radians(120), target=(0, 0, -1))
    return SetupResult(rig, "endoscope", 0.0, 0.06, (4, 7))


# ---------------------------------------------------------------------------
# 5. Cantilever printer in an IKEA Lack enclosure, IR security cam at night
# ---------------------------------------------------------------------------

def palette_lack(rng):
    return {"frame": "black", "accent": (0.4, 0.4, 0.42), "plate": "textured_gold",
            "cover": (0.6, 0.6, 0.62), "body": (0.05, 0.05, 0.055), "infrared": True,
            "dust": 0.3}


def build_lack_ir_night(m, colls, rng, height, clutter=True):
    env, lights = colls["Environment"], colls["Lights"]
    inner, half = 0.48, 0.275
    box("LackBottom", (0, 0, -0.025), (0.55, 0.55, 0.05), m["lack"], env, PASS_ENVIRONMENT)
    box("LackTop", (0, 0, inner + 0.025), (0.55, 0.55, 0.05), m["lack"], env,
        PASS_ENVIRONMENT)
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(f"LackLeg{sx:+d}{sy:+d}", (sx * 0.25, sy * 0.25, inner / 2),
                (0.05, 0.05, inner), m["lack"], env, PASS_ENVIRONMENT)
    for sx in (-1, 1):
        box(f"PanelSide{sx:+d}", (sx * (half + 0.002), 0, inner / 2), (0.003, 0.55, inner),
            m["acrylic"], env, PASS_ENVIRONMENT)
    box("PanelBack", (0, half + 0.002, inner / 2), (0.55, 0.003, inner), m["acrylic"], env,
        PASS_ENVIRONMENT)
    box("PanelFront", (0, -half - 0.002, inner / 2), (0.55, 0.003, inner), m["acrylic"],
        env, PASS_ENVIRONMENT)
    box("Floor", (0, 0, -0.45), (4, 4, 0.02), m["concrete"], env, PASS_ENVIRONMENT)
    box("Wall", (0, 0.6, 0.5), (4, 0.02, 2), m["paint"], env, PASS_ENVIRONMENT)

    rig = build_cantilever(m, colls, height)

    # Pi on the inside of the left panel, ribbon cable and power looms down
    # the back leg, LED strip under the top, blue tape patching the panels.
    cl.raspberry_pi(m, env, rng, (-0.27, 0.12, 0.3), math.pi / 2, vertical=True)
    tube("Loom", [catmull([(0.24, 0.24, 0.47), (0.22, 0.23, 0.3), (0.235, 0.22, 0.1),
                              (0.1, 0.22, 0.0015)], 10)], 0.004, m["cable"], env,
         pass_index=PASS_ENVIRONMENT)
    tube("RibbonToCam", [catmull([(-0.26, 0.12, 0.33), (-0.2, 0.0, 0.47),
                                     (0.1, -0.2, 0.475)], 10)], 0.002, m["cable"], env,
         pass_index=PASS_ENVIRONMENT)
    box("LEDStripOff", (0, -0.2, inner - 0.002), (0.45, 0.01, 0.002), m["pcb"], env,
        PASS_ENVIRONMENT)
    tape = principled("Tape", (0.05, 0.25, 0.65), 0.7)
    for i in range(rng.randint(2, 5)):
        box(f"PanelTape{i}", (rng.uniform(-0.2, 0.2), -half - 0.004, rng.uniform(0.02, 0.46)),
            (0.05, 0.0008, 0.02), tape, env, PASS_ENVIRONMENT,
            rotation=(0, rng.uniform(-1.4, 1.4), 0))
    if clutter:
        floor = (-0.22, 0.22, -0.22, 0.22)
        base = [(-0.2, 0.2, -0.19, 0.19)]
        cl.debris(m, env, rng, floor, 0.0, base, purge=rng.randint(6, 12),
                  offcuts=rng.randint(3, 6), hair=rng.randint(2, 4))
        cl.scraper(m, env, rng, (rng.uniform(-0.1, 0.1), -0.225, 0.0), rng.uniform(-0.3, 0.3))
        cl.hex_keys(m, env, rng, (0.2, -0.21, 0.0), rng.uniform(0, 6.3))
        cl.old_print(m, env, rng, (-0.2, -0.22, 0.0), rng.uniform(0, 6.3))
        cl.spaghetti_ball(env, rng, (0.21, 0.0, 0.0), radius=0.02)
    set_world((0.02, 0.02, 0.03), 0.02)

    cam = build_camera("SecurityCam", "ir_night", colls["CameraRig"],
                       (0.15, -half - 0.03, 0.36), (0.0, -0.03, 0.075))
    box("SecurityCamBody", (0, 0, 0.03), (0.06, 0.06, 0.05), m["lack"], colls["CameraRig"],
        PASS_ENVIRONMENT, bevel=0.01, parent=cam)
    # 850 nm illuminator next to the lens. In a monochrome IR frame only the
    # amount of light matters, so it is modelled as white.
    point_light("IRIlluminator", lights, (0.0, -0.012, 0.0), 10.0, (1.0, 1.0, 1.0),
                radius=0.008, parent=cam, spot=math.radians(95), target=(0, -0.012, -1))
    return SetupResult(rig, "ir_night", 0.0, 0.1)


SETUPS = {
    "enclosed_corexy": (palette_enclosed_corexy, build_enclosed_corexy),
    "prusa_desk": (palette_prusa_desk, build_prusa_desk),
    "voron_topcam": (palette_voron, build_voron_topcam),
    "ender_nozzlecam": (palette_ender, build_ender_nozzlecam),
    "lack_ir_night": (palette_lack, build_lack_ir_night),
}
