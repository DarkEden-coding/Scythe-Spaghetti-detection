"""The parts on the bed: intact prints, and the delamination -> spaghetti
failure.

Everything here is in bed-local coordinates (origin at the centre of the
plate's top surface), parented to the printer rig's `bed` empty, so the same
code works for printers whose bed moves in Z, in Y, or not at all.
"""

import math

import bmesh
import bpy
from mathutils import Vector
from mathutils import noise as mnoise

from .util import (LAYER, LINE_W, PASS_DELAMINATION, PASS_PRINT, PASS_SPAGHETTI,
                   empty, filled_profile, link, noise_vector, rounded_outline, tube)

WALL_LOOPS = 3
WALL = WALL_LOOPS * LINE_W * 0.94


# ---------------------------------------------------------------------------
# Part shapes (2D signed distance fields + matching offset loops)
# ---------------------------------------------------------------------------

class Shape:
    """Common placement, transforms and infill clipping. Subclasses provide
    `sdf`, `outer_loop(inset)` and `hole_loops(outset)` in world (bed) XY."""

    def __init__(self, cx, cy, angle, infill_angle, infill_spacing):
        self.cx, self.cy, self.angle = cx, cy, angle
        self.infill_angle = infill_angle
        self.infill_spacing = infill_spacing

    @property
    def radius(self):
        return math.hypot(self.hx, self.hy)

    def to_world(self, lx, ly):
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        return (self.cx + lx * ca - ly * sa, self.cy + lx * sa + ly * ca)

    def to_local(self, x, y):
        dx, dy = x - self.cx, y - self.cy
        ca, sa = math.cos(-self.angle), math.sin(-self.angle)
        return (dx * ca - dy * sa, dx * sa + dy * ca)

    def _circle(self, lx, ly, r, n):
        return [self.to_world(lx + r * math.cos(2 * math.pi * i / n),
                              ly + r * math.sin(2 * math.pi * i / n)) for i in range(n)]

    def infill_segments(self):
        """Grid infill (two perpendicular families), clipped to the interior."""
        segs = []
        reach = self.radius + 0.002
        for fam in (0, 1):
            ang = self.infill_angle + fam * math.pi / 2
            d = Vector((math.cos(ang), math.sin(ang)))
            nrm = Vector((-d.y, d.x))
            k_max = int(reach / self.infill_spacing)
            for k in range(-k_max, k_max + 1):
                base = Vector((self.cx, self.cy)) + nrm * (k * self.infill_spacing)
                run, steps = None, int(2 * reach / 0.0002)
                for s in range(steps + 1):
                    t = -reach + s * 0.0002
                    p = base + d * t
                    inside = self.sdf(p.x, p.y) < -(WALL - 0.0002)
                    if inside and run is None:
                        run = t
                    elif not inside and run is not None:
                        if t - run > 0.0015:
                            segs.append((fam, k, base + d * run, base + d * t))
                        run = None
        return segs


class Bracket(Shape):
    """Rounded rectangle with two through holes."""

    def __init__(self, rng, scale, **kw):
        super().__init__(**kw)
        self.hx = rng.uniform(0.032, 0.042) * scale
        self.hy = rng.uniform(0.020, 0.026) * scale
        self.r = rng.uniform(0.004, 0.008) * scale
        self.holes = [(-self.hx * 0.45, 0.0, self.hy * 0.42),
                      (self.hx * 0.55, self.hy * 0.3, self.hy * 0.22)]

    def sdf(self, x, y):
        lx, ly = self.to_local(x, y)
        qx = abs(lx) - (self.hx - self.r)
        qy = abs(ly) - (self.hy - self.r)
        d = math.hypot(max(qx, 0), max(qy, 0)) + min(max(qx, qy), 0) - self.r
        for hx, hy, hr in self.holes:
            d = max(d, hr - math.hypot(lx - hx, ly - hy))
        return d

    def outer_loop(self, inset):
        pts = rounded_outline(self.hx - inset, self.hy - inset,
                              max(self.r - inset, 0.0003), n=10)
        return [self.to_world(*p) for p in pts]

    def hole_loops(self, outset):
        return [self._circle(hx, hy, hr + outset, 40) for hx, hy, hr in self.holes]


class Disc(Shape):
    """Round boss / bushing: a disc with a centre bore."""

    def __init__(self, rng, scale, **kw):
        super().__init__(**kw)
        self.R = rng.uniform(0.018, 0.03) * scale
        self.bore = self.R * rng.uniform(0.25, 0.45)
        self.hx = self.hy = self.R

    def sdf(self, x, y):
        lx, ly = self.to_local(x, y)
        d = math.hypot(lx, ly)
        return max(d - self.R, self.bore - d)

    def outer_loop(self, inset):
        return self._circle(0, 0, self.R - inset, 72)

    def hole_loops(self, outset):
        return [self._circle(0, 0, self.bore + outset, 48)]


def random_parts(rng, plate_half, count=None):
    """1-3 non-overlapping parts near the middle of the plate."""
    count = count or rng.choice((1, 1, 2, 3))
    scale = min(1.0, min(plate_half) / 0.12)
    infill_angle = rng.choice((math.radians(45), 0.0))
    spacing = rng.uniform(0.004, 0.0065)
    parts = []
    for _ in range(count * 30):
        if len(parts) == count:
            break
        kind = rng.choice((Bracket, Bracket, Disc))
        reach = 0.25 + 0.2 * (count - 1)
        part = kind(rng, scale, cx=rng.uniform(-1, 1) * plate_half[0] * reach,
                    cy=rng.uniform(-1, 1) * plate_half[1] * reach,
                    angle=math.radians(rng.uniform(-30, 30)),
                    infill_angle=infill_angle, infill_spacing=spacing)
        if all(math.hypot(part.cx - o.cx, part.cy - o.cy) > part.radius + o.radius + 0.006
               for o in parts):
            parts.append(part)
    return parts


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def _xy(loop):
    return [(p[0], p[1]) for p in loop]


def walls_object(name, part, z0, z1, mat, coll, parent, pass_index=PASS_PRINT):
    """Perimeter walls from z0 to z1: outer ring + ring around each hole."""
    loops = [part.outer_loop(0.0), part.outer_loop(WALL)]
    loops += part.hole_loops(WALL) + part.hole_loops(0.0)
    return filled_profile(name, loops, z1 - z0, (0, 0, (z0 + z1) / 2), mat, coll,
                          pass_index, parent)


def bottom_object(name, part, mat, coll, parent):
    """Solid bottom skin."""
    loops = [part.outer_loop(0.0)] + part.hole_loops(0.0)
    return filled_profile(name, loops, 0.0008, (0, 0, 0.0004), mat, coll, PASS_PRINT,
                          parent)


def infill_object(name, segs, z0, z1, mat, coll, parent, pass_index=PASS_PRINT):
    """Sparse infill walls, one thin box per clipped segment."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    for _, _, a, b in segs:
        d = (b - a).normalized()
        n = Vector((-d.y, d.x)) * (LINE_W / 2)
        corners = [a + n, b + n, b - n, a - n]
        vs = [bm.verts.new((p.x, p.y, z)) for z in (z0, z1) for p in corners]
        for f in ((0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2),
                  (2, 6, 7, 3), (3, 7, 4, 0)):
            bm.faces.new([vs[i] for i in f])
    bm.to_mesh(mesh)
    bm.free()
    return link(bpy.data.objects.new(name, mesh), coll, mat, pass_index, parent)


def ordered_infill(segs):
    """Infill lines in print order; the slicer alternates direction."""
    out = []
    for fam, k, a, b in sorted(segs, key=lambda s: (s[0], s[1])):
        out.append((b, a) if k % 2 else (a, b))
    return out


def current_layer(name, part, segs, zc, fraction, m, coll, parent):
    """Top layer in progress: perimeters done, `fraction` of the infill laid.
    Returns the nozzle position and travel direction when fraction < 1."""
    squash = LAYER / LINE_W
    perims = []
    for i in range(WALL_LOOPS):
        off = LINE_W / 2 + i * LINE_W * 0.94
        perims.append([(*p, zc) for p in part.outer_loop(off)])
        perims.extend([(*p, zc) for p in loop] for loop in part.hole_loops(off))
    tube(f"{name}Perimeters", perims, LINE_W / 2, m["pla_top"], coll, cyclic=True,
         pass_index=PASS_PRINT, z_squash=squash, origin=(0, 0, zc), resolution=2,
         parent=parent)

    order = ordered_infill(segs)
    done = int(len(order) * fraction)
    lines = [[(a.x, a.y, zc), (b.x, b.y, zc)] for a, b in order[:done]]
    nozzle = direction = None
    if done < len(order):
        a, b = order[done]
        nozzle = a + (b - a) * 0.5
        direction = (b - a).normalized()
        lines.append([(a.x, a.y, zc), (nozzle.x, nozzle.y, zc)])
    if lines:
        tube(f"{name}Infill", lines, LINE_W / 2, m["pla_top"], coll,
             pass_index=PASS_PRINT, z_squash=squash, origin=(0, 0, zc), resolution=2,
             parent=parent)
    return nozzle, direction


def intact_part(name, part, height, m, coll, bed, fraction):
    top = height - LAYER  # last completed layer
    segs = part.infill_segments()
    bottom_object(f"{name}Bottom", part, m["pla"], coll, bed)
    walls_object(f"{name}Walls", part, 0.0, top, m["pla"], coll, bed)
    infill_object(f"{name}Infill", segs, 0.0008, top, m["pla"], coll, bed)
    if fraction > 0:
        return current_layer(f"{name}Layer", part, segs, top + LAYER / 2, fraction,
                             m, coll, bed)
    return None, None


def purge_line(m, coll, bed, plate_half):
    """Purge line along the front-left edge of the plate, done before the print."""
    py = -plate_half[1] + 0.006
    x0 = -plate_half[0] * 0.92
    purge = [(x, py + 0.0004 * math.sin(x * 900), 0.0003)
             for x in (x0 + 0.002 * i for i in range(55))]
    tube("PurgeLine", [purge], 0.0009, m["pla"], coll, pass_index=PASS_PRINT,
         z_squash=0.45, origin=(0, 0, 0.0003), parent=bed)


# ---------------------------------------------------------------------------
# Failure: delamination followed by spaghetti
# ---------------------------------------------------------------------------

def layer_path(part, segs, rng, infill_fraction):
    """One layer's toolpath as a list of pieces (perimeters, then a random
    subset of infill lines), each a list of 2D Vectors resampled at ~0.5 mm."""
    pieces = [part.outer_loop(LINE_W / 2)] + part.hole_loops(LINE_W / 2)
    pieces = [pts + pts[:1] for pts in pieces]
    for a, b in ordered_infill(segs):
        if rng.random() < infill_fraction:
            pieces.append([tuple(a), tuple(b)])
    out = []
    for pts in pieces:
        res = []
        for p, q in zip(pts, pts[1:]):
            p, q = Vector(p[:2]), Vector(q[:2])
            n = max(1, int((q - p).length / 0.0005))
            res.extend(p.lerp(q, i / n) for i in range(n))
        res.append(Vector(pts[-1][:2]))
        out.append(res)
    return out


def smooth(values, window):
    half = window // 2
    out = []
    for i in range(len(values)):
        chunk = values[max(0, i - half):i + half + 1]
        total = chunk[0]
        for v in chunk[1:]:
            total = total + v
        out.append(total / len(chunk))
    return out


def spaghetti_strands(part, segs, rng, top_height, good, height):
    """Filament extruded into air after the failure, as a list of polylines.

    Each pass replays the layer toolpath at the nozzle height of that moment.
    The extruded strand is not supported, so it drifts (a coherent noise field
    plus drag from the moving nozzle), coils, and settles on whatever is below
    it: the broken part, earlier strands, or the bed. A coarse heightfield
    tracks that surface so the nest builds up and drapes over the edges."""
    radius = 0.00021
    cell = 0.001
    field = {}
    offset = Vector((rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(0, 100)))

    def surface(q):
        key = (round(q.x / cell), round(q.y / cell))
        if key not in field:
            field[key] = top_height(q.x, q.y)
        return key, field[key]

    strands, last, piece = [], None, None
    n_pass = max(8, int((height - good) / 0.0009))
    for j in range(n_pass):
        z_noz = good + (j + 1) / n_pass * (height - good)
        mess = 0.3 + 2.2 * (j / n_pass) ** 1.3
        curl_r = rng.uniform(0.0015, 0.006)
        curl_rate = rng.choice((0.0, 0.5, 1.0, 1.0))
        phase = rng.uniform(0, 2 * math.pi)
        arc = 0.0
        for piece in layer_path(part, segs, rng, infill_fraction=0.2):
            pts = []
            for p in piece:
                arc += 0.0005
                f = noise_vector(Vector((p.x * 22, p.y * 22, j * 0.23)) + offset)
                drift = Vector((f.x, f.y)) * 0.012 * mess
                # Drag from the moving nozzle: smooth along the strand, not per point.
                g = noise_vector(Vector((arc * 70, j * 1.7, 5.0)) + offset)
                drag = Vector((g.x, g.y)) * 0.004 * mess
                phase += 0.0005 / curl_r * curl_rate
                curl = Vector((math.cos(phase), math.sin(phase))) * curl_r * curl_rate
                q = p + drift + drag + curl
                key, surf = surface(q)
                fl = mnoise.noise(Vector((q.x * 90, q.y * 90, j * 0.7)) + offset)
                air = max(0.0, z_noz - surf)
                fluff = (max(0.0, fl) * 1.6) ** 1.5 * min(0.006, air)
                # Coils extruded into air stand up as vertical loops.
                rise = curl_r * curl_rate * (1 + math.sin(phase)) * 0.8
                z_rest = min(surf + radius + fluff, z_noz)
                z = min(z_rest + min(rise, air * 0.6), z_noz)
                pts.append((q, z))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        k2 = (key[0] + dx, key[1] + dy)
                        base = field.get(k2, top_height(k2[0] * cell, k2[1] * cell))
                        lift = z_rest + radius * (0.8 if dx == dy == 0 else 0.3)
                        field[k2] = max(base, lift)
            # Break into strands at travel moves (mostly) and at random snaps.
            if last is not None and rng.random() < 0.35:
                pts = last + pts
                strands.pop()
            chunks, cur = [], []
            for item in pts:
                cur.append(item)
                if len(cur) > 40 and rng.random() < 0.0025:
                    chunks.append(cur)
                    cur = [item]
            chunks.append(cur)
            for ch in chunks:
                if len(ch) < 3:
                    continue
                xy = smooth([q for q, _ in ch], 5)
                zs = smooth([z for _, z in ch], 7)  # sag between supports
                strands.append([(v.x, v.y, max(z, radius)) for v, z in zip(xy, zs)])
            last = pts
    return strands, piece[-1], (piece[-1] - piece[-2]).normalized()


def failed_part(part, height, m, colls, bed, rng):
    """Part prints fine to `good`, the upper layers delaminate at `crack` and
    lift/shift off the base, and everything after that is spaghetti."""
    c, f = colls["Print"], colls["Failure"]
    good = round(height * rng.uniform(0.35, 0.5) / LAYER) * LAYER
    crack = round(good * rng.uniform(0.45, 0.65) / LAYER) * LAYER
    segs = part.infill_segments()

    # Intact base.
    bottom_object("FailedBottom", part, m["pla"], c, bed)
    walls_object("FailedWalls", part, 0.0, crack, m["pla"], c, bed)
    infill_object("FailedInfill", segs, 0.0008, crack, m["pla"], c, bed)

    # Delaminated section, hinged on one long edge so the far edge lifts.
    side = rng.choice((-1, 1))
    tilt = math.radians(rng.uniform(3.0, 5.5))
    shift = rng.uniform(0.0004, 0.0015) * rng.choice((-1, 1))
    hinge = empty("DelaminationHinge", f, (*part.to_world(0.0, -side * part.hy), crack),
                  parent=bed)
    hinge.rotation_euler = (0, 0, part.angle)
    rest = hinge.matrix_basis.inverted()
    upper = [walls_object("DelaminatedWalls", part, crack, good, m["pla"], f, None,
                          PASS_DELAMINATION),
             infill_object("DelaminatedInfill", segs, crack, good, m["pla"], f, None,
                           PASS_DELAMINATION)]
    for obj in upper:
        obj.parent = hinge
        obj.matrix_parent_inverse = rest
    hinge.rotation_euler = (side * tilt, 0, part.angle)
    hinge.location.x += shift * math.cos(part.angle)
    hinge.location.y += shift * math.sin(part.angle)

    def top_height(x, y):
        if part.sdf(x, y) >= 0:
            return 0.0
        _, ly = part.to_local(x, y)
        return good + max(0.0, side * ly + part.hy) * math.tan(tilt)

    strands, nozzle, direction = spaghetti_strands(part, segs, rng, top_height,
                                                   good, height)
    # The live strand: from the nest up to the nozzle tip, hanging slightly.
    end = Vector(strands[-1][-1])
    tip = Vector((nozzle.x, nozzle.y, height - 0.0004))
    hang = [end.lerp(tip, t) - Vector((0, 0, 0.0025 * math.sin(math.pi * t)))
            for t in (i / 12 for i in range(1, 13))]
    strands[-1].extend(tuple(p) for p in hang)
    tube("Spaghetti", strands, 0.00021, m["pla_top"], f, pass_index=PASS_SPAGHETTI,
         resolution=2, parent=bed)

    info = {"good_mm": round(good * 1000, 2), "crack_mm": round(crack * 1000, 2),
            "tilt_deg": round(math.degrees(tilt), 2), "strands": len(strands)}
    return nozzle, direction, info


def nozzle_blob(m, colls, carriage):
    """Lump of filament wrapped around the hotend, typical once spaghetti
    starts curling back up into the nozzle. Carriage origin is the nozzle tip."""
    mesh = bpy.data.meshes.new("NozzleBlob")
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=4, radius=1.0)
    for v in bm.verts:
        d = mnoise.noise(v.co * 2.2 + Vector((3.1, 7.7, 1.3)))
        v.co *= 1.0 + 0.28 * d
        v.co = Vector((v.co.x * 0.0075, v.co.y * 0.0085, v.co.z * 0.0045))
    bm.to_mesh(mesh)
    bm.free()
    mesh.shade_smooth()
    obj = bpy.data.objects.new("NozzleBlob", mesh)
    obj.location = (0.001, 0.002, 0.0045)
    return link(obj, colls["Failure"], m["pla_top"], PASS_SPAGHETTI, carriage)


def build_print(m, colls, rng, rig, height, failure):
    """All parts on the plate. Returns (nozzle_xy, direction, info, parts),
    with the nozzle in bed-local coordinates."""
    bed = rig.bed
    parts = random_parts(rng, rig.plate_half)
    purge_line(m, colls["Print"], bed, rig.plate_half)
    info = {"parts": len(parts)}
    if failure:
        # The first part fails; the rest keep printing normally around it.
        nozzle, direction, extra = failed_part(parts[0], height, m, colls, bed, rng)
        info.update(extra)
        for i, part in enumerate(parts[1:]):
            intact_part(f"Part{i + 1}", part, height, m, colls["Print"], bed, 1.0)
        nozzle_blob(m, colls, rig.carriage)
        return nozzle, direction, info, parts

    # The nozzle is partway through the last part's layer; the other parts
    # have already finished it.
    nozzle = direction = None
    for i, part in enumerate(parts):
        fraction = 1.0 if i < len(parts) - 1 else rng.uniform(0.45, 0.8)
        n, d = intact_part(f"Part{i}", part, height, m, colls["Print"], bed, fraction)
        if n is not None:
            nozzle, direction = n, d
    return nozzle, direction, info, parts
