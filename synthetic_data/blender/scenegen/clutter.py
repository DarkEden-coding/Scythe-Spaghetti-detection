"""Clutter: the stuff that accumulates around real printers.

Purge blobs, filament offcuts, stringing hair, old and failed prints, tools,
cables, a Raspberry Pi, tape, spools. Placement is drawn from the setup's RNG
so every seed gets a different mess.

Loose filament here is deliberately close in appearance to spaghetti (curved
filament-coloured strands) but is not a failure, so a detector cannot learn
"curvy filament == spaghetti" as a shortcut. All clutter is PASS_ENVIRONMENT.

Each prop is built around an empty at `loc` (z = the surface it rests on)
rotated by `rot` about Z, with children in prop-local coordinates.
"""

import math

import bmesh
import bpy
from mathutils import Vector
from mathutils import noise as mnoise

from .materials import principled
from .util import (PASS_ENVIRONMENT, box, catmull, cylinder, empty, filled_profile, link,
                   noise_vector, tube)

P = PASS_ENVIRONMENT

SCRAP_COLORS = [(0.80, 0.20, 0.03), (0.10, 0.10, 0.11), (0.78, 0.78, 0.74),
                (0.02, 0.45, 0.20), (0.60, 0.02, 0.03), (0.03, 0.12, 0.55),
                (0.85, 0.60, 0.02), (0.25, 0.04, 0.45), (0.02, 0.02, 0.02),
                (0.75, 0.75, 0.75)]


def plastic(rng, rgb=None, rough=0.3):
    rgb = rgb or rng.choice(SCRAP_COLORS)
    return principled("Scrap", rgb, rough,
                      **{"Subsurface Weight": 0.15, "Subsurface Radius": (1.0, 0.6, 0.4),
                         "Subsurface Scale": 0.001})


def root(name, c, loc, rot, parent=None):
    r = empty(name, c, loc, parent)
    r.rotation_euler = (0, 0, rot)
    return r


def blob_mesh(name, rng, size, squash=0.6, lumps=0.35, subdiv=3):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
    off = Vector((rng.uniform(0, 50), rng.uniform(0, 50), rng.uniform(0, 50)))
    for v in bm.verts:
        d = mnoise.noise(v.co * 2.0 + off)
        v.co *= 1.0 + lumps * d
        v.co = Vector((v.co.x * size[0], v.co.y * size[1], v.co.z * size[2] * squash))
    bm.to_mesh(mesh)
    bm.free()
    mesh.shade_smooth()
    return mesh


# ---------------------------------------------------------------------------
# Filament debris
# ---------------------------------------------------------------------------

def purge_blob(m, c, rng, loc, parent=None, mat=None):
    """Purge 'poop': a flattened, lumpy coil of extruded filament, often
    striped where the colour changed partway through."""
    s = rng.uniform(0.006, 0.014)
    mesh = blob_mesh("PurgeBlob", rng, (s, s * rng.uniform(0.7, 1.0), s), squash=0.5)
    obj = bpy.data.objects.new("PurgeBlob", mesh)
    obj.location = (loc[0], loc[1], loc[2] + s * 0.25)
    obj.rotation_euler = (rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4),
                          rng.uniform(0, 2 * math.pi))
    link(obj, c, mat or plastic(rng), P, parent)
    # Second colour layered on top: the colour-change transition.
    if rng.random() < 0.6:
        obj.data.materials.append(plastic(rng))
        cut = rng.uniform(-0.3, 0.3) * s
        for poly in obj.data.polygons:
            poly.material_index = 1 if poly.center.z > cut else 0
    return obj


def filament_offcut(c, rng, loc, length=None, mat=None, parent=None, radius=0.000875):
    """A cut-off piece of 1.75 mm filament that kept its spool curl."""
    length = length or rng.uniform(0.03, 0.16)
    curl = rng.uniform(0.03, 0.09)  # spool memory radius
    a0 = rng.uniform(0, 2 * math.pi)
    n = max(6, int(length / 0.004))
    pts = []
    for i in range(n + 1):
        a = a0 + (i / n) * length / curl
        pts.append((loc[0] + curl * (math.cos(a) - math.cos(a0)),
                    loc[1] + curl * (math.sin(a) - math.sin(a0)),
                    loc[2] + radius + 0.0015 * max(0.0, math.sin(i / n * math.pi * 2.3))))
    return tube("Offcut", [pts], radius, mat or plastic(rng), c, pass_index=P, parent=parent,
                resolution=2)


def stringing_hair(c, rng, loc, spread=0.03, mat=None, parent=None):
    """Clump of fine strings (pulled off a print and dropped)."""
    mat = mat or plastic(rng)
    off = Vector((rng.uniform(0, 50), rng.uniform(0, 50), 0))
    splines = []
    for _ in range(rng.randint(4, 10)):
        p = Vector((loc[0] + rng.uniform(-spread, spread) / 2,
                    loc[1] + rng.uniform(-spread, spread) / 2, loc[2]))
        pts = []
        for _ in range(rng.randint(8, 20)):
            f = noise_vector(p * 60 + off)
            jitter = Vector((rng.gauss(0, 1), rng.gauss(0, 1), 0))
            p = p + Vector((f.x, f.y, 0)) * 0.006 + jitter * 0.0015
            pts.append((p.x, p.y, loc[2] + 0.0002 + abs(f.z) * 0.002))
        splines.append(pts)
    return tube("StringingHair", splines, 0.00015, mat, c, pass_index=P, parent=parent,
                resolution=1)


def spaghetti_ball(c, rng, loc, radius=0.035, mat=None, parent=None):
    """A failed print that was pulled off and tossed aside."""
    mat = mat or plastic(rng)
    off = Vector((rng.uniform(0, 50), rng.uniform(0, 50), rng.uniform(0, 50)))
    center = Vector(loc) + Vector((0, 0, radius * 0.6))
    splines = []
    for _ in range(rng.randint(10, 18)):
        p = center + Vector((rng.uniform(-1, 1), rng.uniform(-1, 1),
                             rng.uniform(-0.5, 0.5))) * radius * 0.6
        d = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))).normalized()
        pts = []
        for _ in range(rng.randint(30, 70)):
            f = noise_vector(p * 40 + off)
            d = (d + f * 0.6).normalized()
            p = p + d * 0.003
            rel = p - center
            rel.z /= 0.6
            if rel.length > radius:  # stay inside a squashed ball
                d = (d - rel.normalized() * 1.2).normalized()
            p.z = max(p.z, loc[2] + 0.0004)
            pts.append(tuple(p))
        splines.append(pts)
    return tube("SpaghettiBall", splines, 0.0004, mat, c, pass_index=P, parent=parent,
                resolution=1)


def old_print(m, c, rng, loc, rot, parent=None):
    """A finished part left lying around: block, cylinder or calibration cube."""
    r = root("OldPrint", c, loc, rot, parent)
    mat = plastic(rng, rough=0.4)
    kind = rng.choice(("block", "cyl", "cube", "plate"))
    if kind == "block":
        s = (rng.uniform(0.02, 0.06), rng.uniform(0.015, 0.04), rng.uniform(0.008, 0.025))
        box("OldPrintBlock", (0, 0, s[2] / 2), s, mat, c, P, bevel=0.002, parent=r)
    elif kind == "cyl":
        rad, h = rng.uniform(0.008, 0.025), rng.uniform(0.01, 0.05)
        cylinder("OldPrintCyl", (0, 0, h / 2), rad, h, mat, c, verts=40, pass_index=P,
                 parent=r)
    elif kind == "cube":
        box("CalibrationCube", (0, 0, 0.01), (0.02, 0.02, 0.02), mat, c, P, bevel=0.0008,
            parent=r)
    else:
        box("OldPrintPlate", (0, 0, 0.0015), (rng.uniform(0.04, 0.09),
                                              rng.uniform(0.03, 0.06), 0.003),
            mat, c, P, bevel=0.0008, parent=r)
    return r


# ---------------------------------------------------------------------------
# Tools and supplies
# ---------------------------------------------------------------------------

def flush_cutters(m, c, rng, loc, rot, parent=None):
    r = root("FlushCutters", c, loc, rot, parent)
    grip = principled("CutterGrip", rng.choice([(0.6, 0.03, 0.02), (0.02, 0.15, 0.55),
                                                (0.8, 0.5, 0.02)]), 0.6)
    spread = rng.uniform(0.006, 0.02)
    for s in (-1, 1):
        handle = catmull([(0.0, s * 0.003, 0.005), (-0.04, s * (0.004 + spread * 0.5), 0.005),
                          (-0.09, s * (0.004 + spread), 0.005)], per_seg=6)
        tube(f"CutterHandle{s:+d}", [handle], 0.0045, grip, c, pass_index=P, parent=r)
    box("CutterJaws", (0.012, 0, 0.004), (0.026, 0.009, 0.005), m["steel"], c, P,
        bevel=0.0015, parent=r)
    cylinder("CutterPivot", (0.0, 0, 0.004), 0.004, 0.007, m["steel"], c, pass_index=P,
             parent=r)
    return r


def scraper(m, c, rng, loc, rot, parent=None):
    r = root("Scraper", c, loc, rot, parent)
    box("ScraperHandle", (-0.05, 0, 0.008), (0.1, 0.026, 0.016),
        principled("ScraperHandle", rng.choice([(0.02, 0.02, 0.02), (0.05, 0.2, 0.05)]), 0.5),
        c, P, bevel=0.005, parent=r)
    box("ScraperBlade", (0.03, 0, 0.003), (0.065, 0.04, 0.0008), m["steel"], c, P,
        bevel=0.0004, parent=r, rotation=(0, math.radians(6), 0))
    return r


def hex_keys(m, c, rng, loc, rot, parent=None):
    r = root("HexKeys", c, loc, rot, parent)
    mat = principled("HexKey", (0.04, 0.04, 0.045), 0.35, metallic=0.9)
    for i in range(rng.randint(2, 5)):
        long, short = rng.uniform(0.05, 0.1), rng.uniform(0.015, 0.03)
        a = rng.uniform(0, 2 * math.pi)
        ox, oy = rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02)
        d = Vector((math.cos(a), math.sin(a), 0))
        n = Vector((-d.y, d.x, 0))
        p0 = Vector((ox, oy, 0.0012))
        tube(f"HexKey{i}", [[tuple(p0), tuple(p0 + d * long), tuple(p0 + d * long + n * short)]],
             0.0012 + 0.0004 * (i % 3), mat, c, pass_index=P, parent=r, resolution=1)
    return r


def glue_stick(m, c, rng, loc, rot, parent=None):
    r = root("GlueStick", c, loc, rot, parent)
    body = principled("GlueStick", rng.choice([(0.7, 0.7, 0.72), (0.3, 0.05, 0.45)]), 0.35)
    cylinder("GlueBody", (0, 0, 0.013), 0.013, 0.07, body, c, axis="X", pass_index=P,
             parent=r)
    cylinder("GlueCap", (0.045, 0, 0.0135), 0.0138, 0.022,
             principled("GlueCap", (0.03, 0.1, 0.5), 0.35), c, axis="X", pass_index=P, parent=r)
    return r


def ipa_bottle(m, c, rng, loc, rot, parent=None):
    r = root("IPABottle", c, loc, rot, parent)
    bottle = principled("IPABottle", (0.85, 0.87, 0.88), 0.15,
                        **{"Transmission Weight": 0.6})
    cylinder("IPABody", (0, 0, 0.07), 0.028, 0.14, bottle, c, verts=40, pass_index=P,
             parent=r)
    box("IPALabel", (0, -0.0282, 0.06), (0.035, 0.001, 0.05),
        principled("Label", (0.6, 0.05, 0.05), 0.5), c, P, parent=r)
    cylinder("IPANeck", (0, 0, 0.147), 0.012, 0.014, m["body"], c, pass_index=P, parent=r)
    box("IPATrigger", (0.008, 0, 0.165), (0.04, 0.018, 0.024), m["body"], c, P,
        bevel=0.004, parent=r)
    return r


def tape_roll(m, c, rng, loc, rot, parent=None, upright=False):
    r = root("TapeRoll", c, loc, rot, parent)
    color = rng.choice([(0.05, 0.25, 0.65), (0.75, 0.6, 0.35), (0.6, 0.65, 0.65)])
    n = 40
    outer = [(0.045 * math.cos(2 * math.pi * i / n), 0.045 * math.sin(2 * math.pi * i / n))
             for i in range(n)]
    inner = [(0.036 * math.cos(2 * math.pi * i / n), 0.036 * math.sin(2 * math.pi * i / n))
             for i in range(n)]
    roll = filled_profile("Tape", [outer, inner], 0.024, (0, 0, 0.012),
                          principled("Tape", color, 0.7), c, P, parent=r)
    if upright:
        roll.location = (0, 0, 0.045)
        roll.rotation_euler = (math.pi / 2, 0, 0)
    return r


def sticky_notes(c, rng, loc, rot, parent=None):
    r = root("StickyNotes", c, loc, rot, parent)
    mat = principled("StickyNote", rng.choice([(0.85, 0.75, 0.12), (0.9, 0.4, 0.5)]), 0.8)
    for i in range(rng.randint(1, 3)):
        box(f"StickyNote{i}", (rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03),
                               0.0004 * (i + 1)),
            (0.076, 0.076, 0.0006), mat, c, P, parent=r,
            rotation=(0, 0, rng.uniform(-0.4, 0.4)))
    return r


def paper(c, rng, loc, rot, parent=None):
    r = root("Paper", c, loc, rot, parent)
    mat = principled("Paper", (0.82, 0.82, 0.8), 0.85)
    for i in range(rng.randint(1, 3)):
        box(f"Sheet{i}", (rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), 0.0003 * (i + 1)),
            (0.21, 0.297, 0.0004), mat, c, P, parent=r,
            rotation=(0, 0, rng.uniform(-0.25, 0.25)))
    return r


def zip_ties(c, rng, loc, parent=None):
    mat = principled("ZipTie", rng.choice([(0.8, 0.8, 0.78), (0.02, 0.02, 0.02)]), 0.45)
    for i in range(rng.randint(2, 6)):
        a = rng.uniform(0, 2 * math.pi)
        length = rng.uniform(0.04, 0.12)
        bend = rng.uniform(-0.3, 0.3)
        pts = []
        for k in range(9):
            t = k / 8
            ang = a + bend * t
            pts.append((loc[0] + math.cos(ang) * length * t + rng.uniform(-0.01, 0.01) * (k == 0),
                        loc[1] + math.sin(ang) * length * t, loc[2] + 0.0007))
        tube(f"ZipTie{i}", [pts], 0.0012, mat, c, pass_index=P, parent=parent,
             resolution=1, z_squash=0.4, origin=(0, 0, loc[2] + 0.0007))


def loose_cable(c, rng, start, end, z, radius=0.0022, mat=None, parent=None, slack=0.06):
    """Cable lying on a surface between two points, with a few lazy bends."""
    mat = mat or principled("Cable", rng.choice([(0.015, 0.015, 0.015), (0.75, 0.75, 0.74)]), 0.55)
    a, b = Vector((start[0], start[1], z)), Vector((end[0], end[1], z))
    d = b - a
    n = Vector((-d.y, d.x, 0)).normalized()
    pts = [tuple(a)]
    for k in range(1, 5):
        p = a.lerp(b, k / 5) + n * rng.uniform(-slack, slack)
        pts.append(tuple(p))
    pts.append(tuple(b))
    pts = [(x, y, z + radius) for x, y, _ in catmull(pts, per_seg=10)]
    return tube("Cable", [pts], radius, mat, c, pass_index=P, parent=parent)


def raspberry_pi(m, c, rng, loc, rot, parent=None, vertical=False):
    r = root("RaspberryPi", c, loc, rot, parent)
    if vertical:
        r.rotation_euler = (math.pi / 2, 0, rot)
    pcb = principled("PiPCB", (0.02, 0.22, 0.06), 0.4)
    box("PiBoard", (0, 0, 0.004), (0.085, 0.056, 0.0016), pcb, c, P, parent=r)
    for i, (x, y, sx, sy, sz) in enumerate(((0.032, 0.012, 0.021, 0.015, 0.016),
                                            (0.032, -0.008, 0.021, 0.015, 0.016),
                                            (0.034, -0.026, 0.018, 0.016, 0.013))):
        box(f"PiPort{i}", (x, y, 0.004 + sz / 2), (sx, sy, sz), m["aluminium"], c, P, parent=r)
    box("PiSoC", (-0.01, 0.0, 0.0055), (0.014, 0.014, 0.0014), m["void"], c, P, parent=r)
    box("PiHeader", (-0.008, 0.024, 0.0075), (0.05, 0.005, 0.006), m["void"], c, P, parent=r)
    box("PiLED", (-0.038, 0.02, 0.0052), (0.002, 0.0015, 0.001),
        principled("PiLED", (0.0, 0.0, 0.0), 0.4, **{"Emission Color": (1, 0.1, 0.05, 1),
                                                     "Emission Strength": 40.0}), c, P, parent=r)
    for x in (-0.039, 0.019):
        for y in (-0.024, 0.024):
            cylinder("PiStandoff", (x, y, 0.0015), 0.0025, 0.003, m["steel"], c, verts=8,
                     pass_index=P, parent=r)
    return r


def spool_lying(m, c, rng, loc, rot, parent=None, color=None, empty_spool=False):
    """A spool on its side, sometimes with a loose tail of filament."""
    r = root("SpoolLying", c, loc, rot, parent)
    flange = principled("SpoolFlange", rng.choice([(0.02, 0.02, 0.02), (0.7, 0.7, 0.7),
                                                  (0.35, 0.25, 0.15)]), 0.4)
    fil = plastic(rng, color)
    for i, z in enumerate((0.002, 0.066)):
        cylinder(f"Flange{i}", (0, 0, z), 0.1, 0.004, flange, c, verts=64, pass_index=P,
                 parent=r)
    wind = 0.04 if empty_spool else rng.uniform(0.055, 0.088)
    cylinder("Winding", (0, 0, 0.034), wind, 0.06, fil, c, verts=64, pass_index=P, parent=r)
    cylinder("Hub", (0, 0, 0.034), 0.027, 0.068, m["void"], c, verts=32, pass_index=P, parent=r)
    if rng.random() < 0.6:
        a = rng.uniform(0, 2 * math.pi)
        tail = [(wind * math.cos(a), wind * math.sin(a), 0.05),
                ((wind + 0.03) * math.cos(a + 0.4), (wind + 0.03) * math.sin(a + 0.4), 0.075),
                ((wind + 0.07) * math.cos(a + 0.7), (wind + 0.07) * math.sin(a + 0.7), 0.04),
                ((wind + 0.1) * math.cos(a + 0.9), (wind + 0.1) * math.sin(a + 0.9), 0.001)]
        tube("SpoolTail", [catmull(tail, 8)], 0.000875, fil, c, pass_index=P, parent=r,
             resolution=2)
    return r


def parts_bin(m, c, rng, loc, rot, parent=None):
    """Open tray with small prints and hardware in it."""
    r = root("PartsBin", c, loc, rot, parent)
    mat = plastic(rng, rough=0.5)
    w, d, h, t = 0.14, 0.1, 0.035, 0.002
    box("BinFloor", (0, 0, t / 2), (w, d, t), mat, c, P, parent=r)
    for s in (-1, 1):
        box(f"BinWallX{s:+d}", (s * (w / 2 - t / 2), 0, h / 2), (t, d, h), mat, c, P, parent=r)
        box(f"BinWallY{s:+d}", (0, s * (d / 2 - t / 2), h / 2), (w, t, h), mat, c, P, parent=r)
    for _ in range(rng.randint(3, 7)):
        old_print(m, c, rng, (rng.uniform(-0.05, 0.05), rng.uniform(-0.03, 0.03), t),
                  rng.uniform(0, 6.3), parent=r).scale = (0.5, 0.5, 0.5)
    return r


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def scatter(rng, n, region, avoid=(), spacing=0.05, tries=60):
    """Up to `n` points in region (x0, x1, y0, y1) outside the `avoid`
    rectangles and at least `spacing` apart."""
    pts = []
    for _ in range(n * tries):
        if len(pts) == n:
            break
        x = rng.uniform(region[0], region[1])
        y = rng.uniform(region[2], region[3])
        if any(a[0] <= x <= a[1] and a[2] <= y <= a[3] for a in avoid):
            continue
        if all(math.hypot(x - px, y - py) >= spacing for px, py in pts):
            pts.append((x, y))
    return pts


def debris(m, c, rng, region, z, avoid=(), purge=6, offcuts=4, hair=2, parent=None):
    """Purge blobs, filament offcuts and stringing hair scattered over a floor."""
    for x, y in scatter(rng, purge, region, avoid, spacing=0.012):
        purge_blob(m, c, rng, (x, y, z), parent)
    for x, y in scatter(rng, offcuts, region, avoid, spacing=0.02):
        filament_offcut(c, rng, (x, y, z), parent=parent)
    for x, y in scatter(rng, hair, region, avoid, spacing=0.03):
        stringing_hair(c, rng, (x, y, z), parent=parent)


def desk_clutter(m, c, rng, region, avoid, z=0.0, count=10):
    """Mixed workbench mess: tools, supplies, old prints, a failed print."""
    makers = [flush_cutters, scraper, hex_keys, glue_stick, tape_roll, ipa_bottle,
              old_print, old_print, old_print, parts_bin]
    rng.shuffle(makers)
    spots = scatter(rng, count, region, avoid, spacing=0.07)
    for i, (x, y) in enumerate(spots):
        maker = makers[i % len(makers)]
        maker(m, c, rng, (x, y, z), rng.uniform(0, 2 * math.pi))
    for x, y in scatter(rng, 2, region, avoid, spacing=0.15):
        spaghetti_ball(c, rng, (x, y, z), radius=rng.uniform(0.025, 0.045))
    for x, y in scatter(rng, 2, region, avoid, spacing=0.1):
        sticky_notes(c, rng, (x, y, z), rng.uniform(0, 6.3))
    for x, y in scatter(rng, 1, region, avoid, spacing=0.1):
        zip_ties(c, rng, (x, y, z))
    debris(m, c, rng, region, z, avoid, purge=8, offcuts=7, hair=3)


def plate_debris(m, colls, rng, rig, parts, count=3):
    """Leftovers on the build plate from earlier prints: strings, a purge
    tail, a peeled skirt fragment. Anywhere on the plate clear of the parts."""
    c, bed = colls["Environment"], rig.bed
    hx, hy = rig.plate_half
    placed = 0
    for _ in range(count * 40):
        if placed == count:
            break
        x, y = rng.uniform(-0.9, 0.9) * hx, rng.uniform(-0.9, 0.9) * hy
        if any(part.sdf(x, y) < 0.012 for part in parts):
            continue
        placed += 1
        pick = rng.random()
        if pick < 0.45:
            stringing_hair(c, rng, (x, y, 0.0), spread=0.02, parent=bed)
        elif pick < 0.75:
            purge_blob(m, c, rng, (x, y, 0.0), parent=bed).scale = (0.6, 0.6, 0.6)
        else:
            # Peeled skirt line: a thin loop fragment curling up off the plate.
            r = rng.uniform(0.01, 0.03)
            a0 = rng.uniform(0, 2 * math.pi)
            pts = [(x + r * math.cos(a0 + t * 3.0), y + r * math.sin(a0 + t * 3.0),
                    0.0002 + 0.002 * t * t) for t in (i / 20 for i in range(21))]
            tube("SkirtFragment", [pts], 0.00022, plastic(rng), c, pass_index=P, parent=bed,
                 resolution=1, z_squash=0.5, origin=(0, 0, 0.0002))
