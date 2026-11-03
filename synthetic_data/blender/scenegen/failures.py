"""Failure timelines (CONTRACTS.md §4 `failure`), driven by the G-code timeline.

Every failure is deterministic from `failure.params.seed` and `onset_s`: all
random choices are drawn here at build time, and anything simulated (the
spaghetti nest) is simulated once up to a fixed horizon after onset and then
revealed by deposition time, so keyframes of one job always agree with each
other and do not depend on which keyframes are rendered.

    spaghetti    the failed copy stops growing at onset; everything the nozzle
                 extrudes over it afterwards falls as strands (class 4), and
                 sometimes collects on the nozzle (class 7)
    detached     slide | tip | knockoff | delaminate: the copy (or its upper
                 section) is rigidly moved at onset (class 5); later
                 extrusion over it air-prints into strands and/or a blob
    layer_shift  from the first layer after onset, all geometry (and the
                 nozzle) is offset by 1-8 mm, sometimes twice (class 6 once
                 the first shifted layer is complete, class 3 before)
    nozzle_blob  a lump grows around the heater block (class 7); the layers
                 printed meanwhile come out ragged and under-extruded

`build(failure, ctx)` returns an object with `pieces(t)`, `state(t)` and
`nozzle_offset(t)`, or None for a clean job.
"""

import math
import random

import numpy as np
from mathutils import Vector
from mathutils import noise as mnoise

from . import meshgen as mg
from .gcodeprint import part_pieces
from .util import (PASS_DETACHED, PASS_NOZZLE_BLOB, PASS_PRINT, PASS_SHIFTED, PASS_SPAGHETTI,
                   noise_vector)

STRAND_R = 0.00022
MAX_STRAND_POINTS = 60000
# Strands are simulated up to this long after onset, whatever keyframes the
# job asks for (the sampler puts post windows at most 1500 s after onset);
# later the nest stops growing. The first FINE_POINTS samples are 0.6 mm
# apart, the rest share the remaining budget.
STRAND_HORIZON_S = 1800.0
FINE_POINTS = 30000
FINE_SPACING = 0.6


# IndexOB class that shows each failure type (CONTRACTS.md §5).
FAILURE_CLASS = {"spaghetti": PASS_SPAGHETTI, "detached": PASS_DETACHED,
                 "layer_shift": PASS_SHIFTED, "nozzle_blob": PASS_NOZZLE_BLOB}


def build(failure, ctx):
    if not failure:
        return None
    kind = failure["type"]
    cls = {"spaghetti": Spaghetti, "detached": Detached, "layer_shift": LayerShift,
           "nozzle_blob": NozzleBlob}.get(kind)
    if cls is None:
        raise ValueError(f"unknown failure type {kind}")
    return cls(failure, ctx)


def _hash01(ids, salt):
    """Deterministic per-id pseudo-random numbers in [0, 1)."""
    x = np.sin(np.asarray(ids, np.float64) * 12.9898 + salt * 78.233) * 43758.5453
    return x - np.floor(x)


def _rot(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        r = [[1, 0, 0], [0, c, -s], [0, s, c]]
    elif axis == "y":
        r = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    else:
        r = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return np.array(r, float)


def _about(rot, pivot):
    """4x4 rotation `rot` about point `pivot`."""
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = np.asarray(pivot) - rot @ np.asarray(pivot)
    return m


class Failure:
    def __init__(self, failure, ctx):
        self.ctx = ctx
        self.plan, self.m = ctx["plan"], ctx["m"]
        self.onset = float(failure["onset_s"])
        self.params = dict(failure.get("params") or {})
        self.seed = int(self.params.get("seed", 0))
        self.rng = random.Random(self.seed)
        self.kind = failure["type"]
        # Geometry depends only on the spec, the timeline and t, never on
        # which keyframes are rendered.
        self.t_end = self.onset + STRAND_HORIZON_S
        self.plate_half = ctx["rig"].plate_half
        self.copy = self._failed_copy()
        self.blob = None

    def _failed_copy(self):
        """One object, picked by seed among those still printing at onset."""
        plan = self.plan
        if len(plan.copies) <= 1:
            return 0
        live = sorted({int(c) for c in plan.copy[(plan.model_rows | plan.support_rows)
                                                 & (plan.t1 > self.onset)] if c >= 0})
        return self.rng.choice(live or list(range(len(plan.copies))))

    def others(self):
        return [c for c in range(len(self.plan.copies)) if c != self.copy]

    def common(self, t):
        """Other copies printing normally, plus brim/purge extras."""
        plan, m = self.plan, self.m
        pieces = []
        for c in self.others():
            pieces += part_pieces(plan, c, t, m)
        pieces.append(("Extras", m["pla_top"], PASS_PRINT, plan.extras(t, self.plate_half)))
        return pieces

    def nozzle_offset(self, t):
        return np.zeros(3)

    def post_rows(self):
        plan = self.plan
        return (plan.ext & (plan.copy == self.copy) & (plan.model_rows | plan.support_rows)
                & (plan.t1 > self.onset))

    def extruded_since(self, t):
        """Filament (mm) extruded over the failed copy since onset."""
        plan = self.plan
        rows = np.flatnonzero(self.post_rows() & (plan.t0 < t))
        if len(rows) == 0:
            return 0.0
        e = self.plan.tl.e[rows].astype(np.float64)
        frac = np.clip((t - np.maximum(plan.t0[rows], self.onset))
                       / (plan.t1[rows] - plan.t0[rows]), 0, 1)
        return float((e * frac).sum())

    def blob_pieces(self, t, growth):
        if self.blob is None or growth <= 0:
            return []
        v, f = self.blob
        # Starts as a lump at the nozzle tip and grows up around the heater
        # block; its underside stays at the tip, where it drags on the print.
        v = v * (0.2 + 0.8 * growth)
        v[:, 2] += -0.0003 - v[:, 2].min()
        return [("NozzleBlob", self.m["pla_top"], PASS_NOZZLE_BLOB, (v, f), "tool")]

    def make_blob(self, rmax):
        rng = self.rng
        # Flattened: it grows sideways under and around the cooling duct, which
        # is what makes it visible from above.
        size = np.array([rmax * rng.uniform(0.9, 1.25), rmax * rng.uniform(0.95, 1.3),
                         rmax * rng.uniform(0.4, 0.6)])
        # Biased forward and to one side, so it bulges out from under the
        # part-cooling shroud as real blobs do.
        centre = (rng.uniform(-0.4, 0.4) * rmax, -rng.uniform(0.1, 0.5) * rmax, 0.0)
        self.blob = mg.blob(centre, size, self.seed, detail=3, lumps=0.45)

    # -- spaghetti strands ------------------------------------------------------

    def height_grid(self, samples):
        """1 mm heightfield (bed-local metres) seeded with `samples` (N, 3)."""
        self.cell = 0.001
        hx, hy = self.plate_half
        self.gx0, self.gy0 = -hx - 0.02, -hy - 0.02
        self.gw = int((2 * hx + 0.04) / self.cell) + 1
        self.gh = int((2 * hy + 0.04) / self.cell) + 1
        grid = np.zeros((self.gw, self.gh))
        if len(samples):
            i = np.clip(((samples[:, 0] - self.gx0) / self.cell).astype(int), 0, self.gw - 1)
            j = np.clip(((samples[:, 1] - self.gy0) / self.cell).astype(int), 0, self.gh - 1)
            np.maximum.at(grid, (i, j), samples[:, 2])
        return grid

    def part_samples(self, t, c, matrix=None):
        """Points on the top surfaces of copy c's rows completed by t (metres)."""
        plan = self.plan
        rows = np.flatnonzero((plan.model_rows | plan.support_rows) & (plan.copy == c)
                              & (plan.t1 <= t))
        if len(rows) == 0:
            return np.zeros((0, 3))
        a, b = plan.p0[rows], plan.p1[rows]
        n = np.maximum(1, (np.linalg.norm(b - a, axis=1) / 0.8).astype(int))
        rep = np.repeat(np.arange(len(rows)), n)
        f = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 0.5) / np.repeat(n, n)
        pts = plan.to_bed(a[rep] + (b[rep] - a[rep]) * f[:, None])
        if matrix is not None:
            pts = mg.transform(pts, matrix)
        return pts

    def simulate_strands(self, grid):
        """Filament extruded over the failed copy after onset, as strands.

        Replays the real toolpath after onset (at the nozzle height of each
        moment). The unsupported strand drifts in a coherent noise field, is
        dragged by the nozzle, coils, and settles on the heightfield, which
        grows as strands pile up. Stores point positions, deposition times
        and strand breaks; `strand_pieces(t)` reveals them by time."""
        plan = self.plan
        # Own stream, so draws made after the simulation do not depend on
        # how many points it took.
        rng = random.Random(self.seed * 7919 + 17)
        rows = np.flatnonzero(self.post_rows() & (plan.t0 < self.t_end))
        if len(rows) == 0:
            self.pts = np.zeros((0, 3))
            self.ptime = np.zeros(0)
            self.brk = np.zeros(0, bool)
            self.prad = np.zeros(0)
            return
        ta = np.maximum(plan.t0[rows], self.onset)
        tb = np.minimum(plan.t1[rows], self.t_end)
        fa = (ta - plan.t0[rows]) / (plan.t1[rows] - plan.t0[rows])
        fb = (tb - plan.t0[rows]) / (plan.t1[rows] - plan.t0[rows])
        a = plan.p0[rows] + (plan.p1[rows] - plan.p0[rows]) * fa[:, None]
        b = plan.p0[rows] + (plan.p1[rows] - plan.p0[rows]) * fb[:, None]
        length = np.linalg.norm(b - a, axis=1)
        fine_len = FINE_SPACING * FINE_POINTS
        coarse = max(FINE_SPACING, (length.sum() - fine_len) / (MAX_STRAND_POINTS - FINE_POINTS))
        spacing = np.where(np.cumsum(length) - length < fine_len, FINE_SPACING, coarse)
        n = np.maximum(1, np.ceil(length / spacing).astype(int))
        rep = np.repeat(np.arange(len(rows)), n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        f = (k + 1) / np.repeat(n, n)
        P = plan.to_bed(a[rep] + (b[rep] - a[rep]) * f[:, None])
        T = ta[rep] + (tb - ta)[rep] * f
        step_m = (length / n)[rep] * 1e-3
        # A new piece starts where consecutive samples are not on adjacent rows.
        newpiece = np.zeros(len(P), bool)
        newpiece[0] = True
        newpiece[1:] = (k[1:] == 0) & ((rows[rep[1:]] != rows[rep[:-1]] + 1)
                                       | (np.linalg.norm(P[1:] - P[:-1], axis=1) > 0.002))

        cell, gx0, gy0, gw, gh = self.cell, self.gx0, self.gy0, self.gw, self.gh
        H = grid.ravel().tolist()
        off = Vector((rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(0, 100)))
        out = np.empty((len(P), 3))
        rad = np.empty(len(P))
        brk = np.zeros(len(P), bool)
        coil, curl_rate, phase, tilt, tc, ts, width = 1.0, 1.0, 0.0, 0.5, 1.0, 0.0, STRAND_R
        arc, run = 0.0, 0
        ramp = float(self.params.get("_mess_ramp_s", 400.0))
        for i in range(len(P)):
            step = step_m[i]
            px, py, zn = P[i]
            if newpiece[i]:
                # Filament falling from a height coils like a viscous rope;
                # each piece gets its own coil size, lean and die swell.
                coil = rng.uniform(0.6, 1.6)
                curl_rate = rng.choice((0.3, 1.0, 1.0, 1.0))
                phase = rng.uniform(0, 2 * math.pi)
                tilt = rng.uniform(0.15, 1.3)
                heading = rng.uniform(0, 2 * math.pi)
                tc, ts = math.cos(heading), math.sin(heading)
                width = STRAND_R * rng.uniform(0.85, 1.5)
                brk[i] = i == 0 or rng.random() < 0.65
                run = 0
            elif run > 40 and rng.random() < 0.0025:
                brk[i] = True
                run = 0
            run += 1
            r = width
            age = min(1.0, (T[i] - self.onset) / ramp)
            mess = 0.3 + 2.2 * age ** 1.3
            arc += step
            ni = min(max(int((px - gx0) / cell), 0), gw - 1)
            nj = min(max(int((py - gy0) / cell), 0), gh - 1)
            fall = max(0.0, zn - H[ni * gh + nj])
            # Coil radius grows with the fall height (1-7 mm).
            R = coil * min(0.007, 0.0012 + 0.18 * fall) * curl_rate
            phase += step / max(R, 1e-4)
            cx, cy = math.cos(phase) * R, math.sin(phase) * R * math.cos(tilt)
            loop_z = (1 + math.sin(phase)) * R * math.sin(tilt)
            fv = noise_vector(Vector((px * 22, py * 22, T[i] * 0.004)) + off)
            gv = noise_vector(Vector((arc * 70, T[i] * 0.03, 5.0)) + off)
            qx = px + fv.x * 0.006 * mess + gv.x * 0.003 * mess + cx * tc - cy * ts
            qy = py + fv.y * 0.006 * mess + gv.y * 0.003 * mess + cx * ts + cy * tc
            ci = min(max(int((qx - gx0) / cell), 0), gw - 1)
            cj = min(max(int((qy - gy0) / cell), 0), gh - 1)
            surf = H[ci * gh + cj]
            air = max(0.0, zn - surf)
            fl = mnoise.noise(Vector((qx * 90, qy * 90, T[i] * 0.01)) + off)
            fluff = (max(0.0, fl) * 1.6) ** 1.5 * min(0.006, air)
            z_rest = min(surf + r + fluff, zn)
            # Cooled strands are stiff and rest on each other loosely: each
            # one wanders up and down through the space between the nest and
            # the nozzle instead of lying flat, so the nest stays see-through.
            wv = mnoise.noise(Vector((arc * 45, T[i] * 0.002, 9.0)) + off)
            loft = (0.5 + 0.5 * wv) ** 2 * air * 0.9
            z = min(z_rest + min(loop_z + loft, air * 0.95), zn)
            out[i] = (qx, qy, max(z, r))
            rad[i] = r
            # Only the lower part of each strand supports the next ones.
            top = z_rest + 0.12 * (z - z_rest)
            for dx in (-1, 0, 1):
                ii = ci + dx
                if 0 <= ii < gw:
                    for dy in (-1, 0, 1):
                        jj = cj + dy
                        if 0 <= jj < gh:
                            lift = top + r * (0.8 if dx == dy == 0 else 0.3)
                            if H[ii * gh + jj] < lift:
                                H[ii * gh + jj] = lift
        # Smooth each strand (sag between contacts), lightly so coils survive.
        starts = np.flatnonzero(brk)
        ends = np.r_[starts[1:], len(out)]
        for s0, s1 in zip(starts, ends):
            if s1 - s0 >= 5:
                for col, win in ((0, 3), (1, 3), (2, 5)):
                    seg = out[s0:s1, col]
                    pad = np.pad(seg, win // 2, mode="edge")
                    out[s0:s1, col] = np.convolve(pad, np.ones(win) / win, mode="valid")
        out[:, 2] = np.maximum(out[:, 2], rad)
        self.pts, self.ptime, self.brk, self.prad = out, T, brk, rad

    def strand_pieces(self, t):
        n = int(np.searchsorted(self.ptime, t, side="right"))
        if n < 2:
            return []
        pts, brk, rad = self.pts[:n], self.brk[:n], self.prad[:n]
        # The live strand hangs from the nest up to the nozzle tip.
        plan = self.plan
        last = plan.rows_done(t, self.post_rows())[1]
        if last is not None:
            tip = plan.nozzle(t) + self.nozzle_offset(t)
            end = pts[-1]
            u = np.linspace(0, 1, 10)[1:, None]
            hang = end + (tip - np.array([0, 0, 0.0004]) - end) * u
            hang[:, 2] -= 0.0025 * np.sin(np.pi * u[:, 0])
            hang[:, 2] = np.maximum(hang[:, 2], STRAND_R)
            pts = np.concatenate([pts, hang])
            brk = np.r_[brk, np.zeros(len(hang), bool)]
            rad = np.r_[rad, np.full(len(hang), rad[-1])]
        geom = mg.tubes(pts, brk, rad, sides=5)
        return [("Spaghetti", self.m["strand"], PASS_SPAGHETTI, geom, "fail")]


# ---------------------------------------------------------------------------

class Spaghetti(Failure):
    """The failed copy stops growing at onset (it lost the plot: a broken
    overhang, a part that started to lift); the nozzle keeps printing into
    air over it and the strands pile up into a nest."""

    def __init__(self, failure, ctx):
        super().__init__(failure, ctx)
        rng = self.rng
        self.params.setdefault("_mess_ramp_s", rng.uniform(200, 600))
        self.e_full = rng.uniform(250, 600)
        samples = np.concatenate([self.part_samples(self.onset, c)
                                  for c in range(len(self.plan.copies))] or [np.zeros((0, 3))])
        self.simulate_strands(self.height_grid(samples))
        self.blob_growth_s = rng.uniform(300, 1200) if rng.random() < 0.4 else None
        if self.blob_growth_s:
            self.make_blob(rng.uniform(0.01, 0.018))

    def pieces(self, t):
        pieces = self.common(t)
        pieces += part_pieces(self.plan, self.copy, t, self.m, freeze=self.onset)
        if t > self.onset:
            pieces += self.strand_pieces(t)
            if self.blob_growth_s:
                pieces += self.blob_pieces(t, min(1.0, (t - self.onset) / self.blob_growth_s))
        return pieces

    def state(self, t):
        e = self.extruded_since(t) if t > self.onset else 0.0
        return {"type": "spaghetti", "active": t >= self.onset,
                "severity": round(float(np.clip(e / self.e_full, 0, 1)), 4),
                "visible": e > 15.0, "extruded_since_onset_mm": round(e, 2)}


class Detached(Failure):
    MODES = ("slide", "tip", "knockoff", "delaminate")

    def __init__(self, failure, ctx):
        super().__init__(failure, ctx)
        rng, plan = self.rng, self.plan
        self.mode = self.params.get("mode") or rng.choice(self.MODES)
        if self.mode not in self.MODES:
            raise ValueError(f"unknown detached mode {self.mode}")
        c = self.copy
        top = plan.top_z(self.onset, c)
        self.top = top
        self.crack = 0.0
        if self.mode == "delaminate":
            lo = plan.level_below(top * rng.uniform(0.3, 0.75))
            self.crack = max(lo, plan.levels[0])
        samples = self.part_samples(self.onset, c)
        self.samples = samples
        # The moved part must land on free plate: the other copies keep
        # printing, and a part pushed into a neighbour is hidden inside it.
        # Draw candidates from the seed until one lands clear (or keep the
        # least-overlapping one).
        best = None
        sides = list("xXyY")
        rng.shuffle(sides)
        topple = None
        for attempt in range(24):
            if self.mode == "tip" and topple and attempt == 8:
                topple = False  # no clear side to fall to: lean instead
            M, severity, topple = self._candidate(attempt, sides[attempt % 4], topple)
            score = self._overlap(M)
            if best is None or score < best[0]:
                best = (score, M, severity)
            if score <= 0.002:
                break
        _, M, self.severity = best
        self.M = M
        # Afterwards the toolpath continues over empty space or the moved part.
        rest = self.part_samples(self.onset, c, M)
        if self.mode == "delaminate":
            base = samples[samples[:, 2] <= self.crack * 1e-3 + 1e-6]
            upper = samples[samples[:, 2] > self.crack * 1e-3 + 1e-6]
            rest = np.concatenate([base, mg.transform(upper, M)])
        others = [self.part_samples(self.onset, k) for k in self.others()]
        self.post = rng.choices(("strands", "blob", "both"), (0.6, 0.15, 0.25))[0]
        self.params.setdefault("_mess_ramp_s", rng.uniform(150, 500))
        if self.post in ("strands", "both"):
            self.simulate_strands(self.height_grid(np.concatenate([rest, *others])))
        if self.post in ("blob", "both"):
            self.blob_growth_s = rng.uniform(200, 900)
            self.make_blob(rng.uniform(0.01, 0.018))

    def _candidate(self, attempt, side, topple):
        """One draw of the rigid motion for the current mode, rested on the
        bed (or on the base below the crack). Returns (M, severity, topple)."""
        rng, plan, c = self.rng, self.plan, self.copy
        box = plan.footprint(c)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        h = self.top * 1e-3
        edge = {"x": (box[0], cy), "X": (box[2], cy), "y": (cx, box[1]), "Y": (cx, box[3])}[side]
        sign = 1 if side.isupper() else -1
        axis = "y" if side.lower() == "x" else "x"

        def tilt(angle, z=0.0):
            # Positive angle leans the top towards the chosen edge.
            a = angle * sign if axis == "y" else -angle * sign
            return _about(_rot(axis, a), (edge[0], edge[1], z))

        if self.mode == "slide":
            d = rng.uniform(0.006, 0.035)
            ang = rng.uniform(0, 2 * math.pi)
            M = _about(_rot("z", math.radians(rng.uniform(-25, 25))), (cx, cy, 0))
            M[:2, 3] += (d * math.cos(ang), d * math.sin(ang))
            severity = min(1.0, d / 0.015)
        elif self.mode == "tip":
            if topple is None:
                topple = rng.random() < 0.5 or h > 0.6 * min(box[2] - box[0], box[3] - box[1])
            angle = math.radians(rng.uniform(80, 95) if topple else rng.uniform(8, 25))
            M = tilt(angle)
            severity = min(1.0, math.degrees(angle) / 20)
        elif self.mode == "knockoff":
            M = _about(_rot("z", rng.uniform(0, 2 * math.pi)), (cx, cy, 0))
            if rng.random() < 0.5:
                M = tilt(math.radians(rng.uniform(85, 95))) @ M
            hx, hy = self.plate_half
            dx = rng.uniform(-hx * 0.75, hx * 0.75) - cx
            dy = rng.uniform(-hy * 0.75, hy * 0.75) - cy
            if math.hypot(dx, dy) < 0.04:
                dx, dy = dx + math.copysign(0.05, dx), dy
            M[:2, 3] += (dx, dy)
            severity = 1.0
        else:  # delaminate
            angle = math.radians(rng.uniform(3, 12))
            M = tilt(angle, self.crack * 1e-3)
            shift = rng.uniform(0, 0.0015)
            ang = rng.uniform(0, 2 * math.pi)
            M[:2, 3] += (shift * math.cos(ang), shift * math.sin(ang))
            severity = min(1.0, math.degrees(angle) / 8)
        samples = self.samples
        if self.mode != "delaminate" and len(samples):
            moved = mg.transform(np.concatenate([samples, samples * [1, 1, 0]]), M)
            M[2, 3] -= moved[:, 2].min()
        return M, severity, topple

    def _overlap(self, M):
        """Fraction of the moved part's samples that land inside another
        copy's footprint (+2 mm) or off the plate."""
        pts = self.samples
        if self.mode == "delaminate":
            pts = pts[pts[:, 2] > self.crack * 1e-3 + 1e-6]
        if len(pts) == 0:
            return 0.0
        pts = pts[:: max(1, len(pts) // 4000)]
        q = mg.transform(np.concatenate([pts, pts * [1, 1, 0]]), M)[:, :2]
        hx, hy = self.plate_half
        bad = (np.abs(q[:, 0]) > hx - 0.002) | (np.abs(q[:, 1]) > hy - 0.002)
        for k in self.others():
            b = self.plan.footprint(k, 0.002)
            bad |= (q[:, 0] > b[0]) & (q[:, 0] < b[2]) & (q[:, 1] > b[1]) & (q[:, 1] < b[3])
        return float(bad.mean())

    def pieces(self, t):
        pieces = self.common(t)
        plan, m, c = self.plan, self.m, self.copy
        if t < self.onset:
            return pieces + part_pieces(plan, c, t, m)
        if self.mode == "delaminate":
            pieces += part_pieces(plan, c, t, m, freeze=self.onset, zr=(-1.0, self.crack))
            pieces += part_pieces(plan, c, t, m, freeze=self.onset, zr=(self.crack, 1e9),
                                  matrix=self.M, pass_index=PASS_DETACHED, prefix="Detached")
        else:
            pieces += part_pieces(plan, c, t, m, freeze=self.onset, matrix=self.M,
                                  pass_index=PASS_DETACHED, prefix="Detached")
        if self.post in ("strands", "both"):
            pieces += self.strand_pieces(t)
        if self.post in ("blob", "both"):
            pieces += self.blob_pieces(t, min(1.0, (t - self.onset) / self.blob_growth_s))
        return pieces

    def state(self, t):
        active = t >= self.onset
        return {"type": "detached", "mode": self.mode, "active": active,
                "severity": round(self.severity if active else 0.0, 4), "visible": active,
                "post": self.post,
                "extruded_since_onset_mm": round(self.extruded_since(t), 2) if active else 0.0}


class LayerShift(Failure):
    def __init__(self, failure, ctx):
        super().__init__(failure, ctx)
        rng, plan, tl = self.rng, self.plan, ctx["tl"]
        self.shifts = []  # (time, layer z top before shift (mm), cumulative offset m)
        total = np.zeros(3)
        t = self.onset
        for k in range(2 if rng.random() < 0.3 else 1):
            if k:
                t = t + rng.uniform(60, 600)
            ts, zb = self._layer_start(t)
            if ts is None:
                break
            mag = rng.uniform(0.001, 0.008)
            if rng.random() < 0.8:  # a skipped X or Y motor step
                ang = rng.choice((0, 0.5, 1, 1.5)) * math.pi
            else:
                ang = rng.uniform(0, 2 * math.pi)
            total = total + np.array([mag * math.cos(ang), mag * math.sin(ang), 0.0])
            self.shifts.append((ts, zb, total.copy()))
        self.tl = tl
        # The first shifted layer completes at done_t; until then its beads
        # look like (and are labelled as) normal print.
        self.done_t = np.inf
        if self.shifts:
            ts = self.shifts[0][0]
            rows = np.flatnonzero(plan.model_rows & (plan.t0 >= ts))
            lay = tl.layer
            self.done_t = (float(plan.t1[rows[lay[rows] == lay[rows[0]]]].max())
                           if len(rows) else ts)

    def _layer_start(self, t):
        """Start time of the first layer starting at or after t, and the top Z
        (mm) of the layer before it."""
        tl, plan = self.ctx["tl"], self.plan
        model = np.flatnonzero(plan.model_rows & (plan.t0 >= t))
        if len(model) == 0:
            return None, None
        layer = tl.layer[model[0]]
        rows = np.flatnonzero(plan.model_rows & (tl.layer == layer))
        ts = float(plan.t0[rows].min())
        if ts < t:  # onset fell inside this layer: shift from the next one
            nxt = np.flatnonzero(plan.model_rows & (tl.layer > layer) & (plan.t0 >= t))
            if len(nxt) == 0:
                return None, None
            layer = tl.layer[nxt[0]]
            ts = float(plan.t0[np.flatnonzero(plan.model_rows & (tl.layer == layer))].min())
        z = float(plan.p1[np.flatnonzero(plan.model_rows & (tl.layer == layer)), 2].min())
        return ts, plan.level_below(z)

    def offset_at(self, t):
        off = np.zeros(3)
        for ts, _, total in self.shifts:
            if t >= ts:
                off = total
        return off

    def nozzle_offset(self, t):
        return self.offset_at(t)

    def pieces(self, t):
        plan, m = self.plan, self.m
        pieces = [("Extras", m["pla_top"], PASS_PRINT, plan.extras(t, self.plate_half))]
        bounds = [(-np.inf, -1.0, np.zeros(3))] + list(self.shifts)
        for c in range(len(plan.copies)):
            for k, (ts, zb, off) in enumerate(bounds):
                tn, zn = (bounds[k + 1][0], bounds[k + 1][1]) if k + 1 < len(bounds) else (np.inf, 1e9)
                if t < ts:
                    break
                shifted = k > 0
                label = shifted and t >= self.done_t
                pieces += part_pieces(plan, c, t, m, zr=(zb, zn), rows_zr=(-1.0, 1e9),
                                      rows_tr=(ts, tn), offset=off if shifted else None,
                                      pass_index=PASS_SHIFTED if label else PASS_PRINT,
                                      prefix="Shifted" if label else "Print")
        return pieces

    def state(self, t):
        if not self.shifts:
            return {"type": "layer_shift", "active": False, "severity": 0.0, "visible": False}
        ts = self.shifts[0][0]
        active = t >= ts
        mag = float(np.linalg.norm(self.offset_at(t))) * 1000
        # The shift is visible (and labelled class 6) once the shifted layer
        # is complete.
        printed = t >= self.done_t
        return {"type": "layer_shift", "active": active,
                "severity": round(float(np.clip(mag / 8.0, 0, 1)) if active else 0.0, 4),
                "visible": active and printed, "shift_layer_printed": printed,
                "shift_mm": [round(float(v) * 1000, 3) for v in self.offset_at(t)[:2]]}


class NozzleBlob(Failure):
    def __init__(self, failure, ctx):
        super().__init__(failure, ctx)
        rng = self.rng
        self.growth_s = rng.uniform(240, 1200)
        self.make_blob(rng.uniform(0.016, 0.026))
        self.drop = rng.uniform(0.3, 0.7)
        self.wobble = rng.uniform(0.3, 0.9)

    def growth(self, t):
        return float(np.clip((t - self.onset) / self.growth_s, 0, 1)) ** 0.7

    def perturb(self, ids, p0, p1, w, h):
        """Rows printed after onset come out ragged: some are missing (the
        filament went into the blob), others wobble and are thin."""
        plan = self.plan
        t0 = plan.t0[ids]
        g = np.clip((t0 - self.onset) / self.growth_s, 0, 1) ** 0.7
        g = np.where(t0 >= self.onset, g, 0.0)
        keep = _hash01(ids, self.seed) >= g * self.drop
        jit = self.wobble * (0.2 + g)
        d0 = (np.stack([_hash01(ids, self.seed + 1), _hash01(ids, self.seed + 2)], 1) - 0.5) * 2
        d1 = (np.stack([_hash01(ids + 1, self.seed + 1), _hash01(ids + 1, self.seed + 2)], 1)
              - 0.5) * 2
        p0, p1 = p0.copy(), p1.copy()
        p0[:, :2] += d0 * (jit * (g > 0))[:, None]
        p1[:, :2] += d1 * (jit * (g > 0))[:, None]
        p1[:, 2] += (_hash01(ids, self.seed + 3) - 0.5) * 0.3 * g
        w = w * (1 - 0.4 * g * _hash01(ids, self.seed + 4))
        return p0[keep], p1[keep], w[keep], h[keep]

    def pieces(self, t):
        plan, m = self.plan, self.m
        pieces = [("Extras", m["pla_top"], PASS_PRINT, plan.extras(t, self.plate_half))]
        for c in range(len(plan.copies)):
            pieces += part_pieces(plan, c, t, m, body_t=self.onset, perturb=self.perturb)
        pieces += self.blob_pieces(t, self.growth(t))
        return pieces

    def state(self, t):
        g = self.growth(t) if t >= self.onset else 0.0
        return {"type": "nozzle_blob", "active": t >= self.onset, "severity": round(g, 4),
                "visible": g > 0.15}
