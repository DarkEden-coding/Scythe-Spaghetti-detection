"""Print geometry at any time t, from a G-code timeline plus the source STL.

At time t each copy on the plate is drawn as:

- the STL clipped below `cut_z` (the top of the layer N_TOP layers under the
  current one), with caps, using the layer-line material;
- every extrusion above `cut_z` up to the nozzle, including the partial move,
  as real toolpath beads;
- supports for all completed layers as beads (collinear runs merged and,
  over budget, every k-th layer with thicker beads);
- brim, skirt and purge lines as beads.

`PrintPlan` precomputes everything that does not depend on t (bed-local
metre coordinates, copies, STL placement, decimated supports) once per job.
All query methods return plain numpy geometry (see meshgen.py); the scene
side (`PrintObjects`) owns the Blender objects and swaps their meshes.
"""

import math

import bpy
import numpy as np

from . import meshgen as mg
from .util import PASS_PRINT

MODEL_KINDS = (1, 2, 3, 4, 5)
SUPPORT_KINDS = (6, 7)
WALL_KINDS = (1, 2)
N_TOP = 3
SUPPORT_BUDGET = 60000     # bead segments for the support history
TOP_BUDGET = 250000        # bead segments drawn as live toolpath
STL_FACE_BUDGET = 200000
ALIGN_MIN_FIT = 0.5        # below this the STL body does not match the toolpath


class PrintPlan:
    def __init__(self, tl, stl_path, scale=1.0, copies=None, log=print):
        self.tl, self.log = tl, log
        meta = tl.meta
        self.center = np.array([*meta["plate_center_mm"], 0.0])
        self.p0 = tl.p0.astype(np.float64)
        self.p1 = tl.p1.astype(np.float64)
        self.kind = tl.kind
        seg = np.linalg.norm(self.p1 - self.p0, axis=1)
        self.ext = (tl.e > 0) & (seg > 1e-4)
        self.t0, self.t1 = tl.t0, tl.t1

        # Bead height: the row's own value when sane, else the layer step.
        lz = np.asarray(tl.layer_z, float)
        step = np.clip(np.diff(np.r_[0.0, lz]), 0.05, 0.5)
        if len(step):
            step[0] = np.clip(lz[0], 0.08, 0.5)
        lay = np.clip(tl.layer, 0, max(len(lz) - 1, 0))
        h = tl.height.astype(np.float64)
        fallback = step[lay] if len(step) else np.full(len(h), 0.2)
        self.h = np.where((h >= 0.05) & (h <= 0.6), h, fallback)
        self.w = np.clip(tl.width.astype(np.float64), 0.25, 2.5)
        self.layer_h = float(np.median(self.h[self.ext & np.isin(self.kind, MODEL_KINDS)])
                             if np.any(self.ext & np.isin(self.kind, MODEL_KINDS)) else 0.2)

        model = self.ext & np.isin(self.kind, MODEL_KINDS)
        self.model_rows = model
        self.support_rows = self.ext & np.isin(self.kind, SUPPORT_KINDS)
        self.levels = np.unique(np.round(self.p1[model, 2], 3)) if model.any() else np.zeros(1)

        self.expected_copies = copies
        objs = getattr(tl, "obj", None)
        self.labelled = bool(getattr(tl, "objects", [])) and objs is not None and \
            bool(np.any(objs[self.model_rows] >= 0))
        if not self.labelled:
            self._find_copies(copies)
        self._place_stl(stl_path, scale)
        self._decimate_supports()
        self._slab_cache = {}

    # -- coordinates -------------------------------------------------------

    def to_bed(self, p_mm):
        return (np.asarray(p_mm, float) - self.center) * 1e-3

    def nozzle(self, t):
        p, _ = self.tl.nozzle_at(t)
        return self.to_bed(p)

    def velocity(self, t):
        return np.asarray(self.tl.velocity_at(t), float) * 1e-3

    # -- copies and STL placement ----------------------------------------------

    def _sample(self, rows, step=1.0, cap=300000):
        """Points (mm) every `step` along the given rows, at most ~`cap`."""
        if len(rows) > cap // 2:
            rows = rows[::int(math.ceil(len(rows) / (cap // 2)))]
        a, b = self.p0[rows], self.p1[rows]
        length = np.linalg.norm(b[:, :2] - a[:, :2], axis=1)
        step = max(step, length.sum() / cap)
        n = np.maximum(1, np.ceil(length / step).astype(int))
        rep = np.repeat(np.arange(len(rows)), n)
        f = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 0.5) / np.repeat(n, n)
        return a[rep] + (b[rep] - a[rep]) * f[:, None]

    def _find_copies(self, copies):
        """Islands: connected components of the wall footprint (1 mm grid,
        dilated 2 mm). `_place_stl` groups or splits them into copies."""
        walls = np.flatnonzero(self.ext & np.isin(self.kind, WALL_KINDS))
        if len(walls) == 0:
            walls = np.flatnonzero(self.model_rows)
        pts = self._sample(walls)[:, :2]
        origin = np.floor(pts.min(axis=0)) - 4 if len(pts) else np.zeros(2)
        ij = np.floor(pts - origin).astype(int)
        shape = ij.max(axis=0) + 5 if len(ij) else np.array([1, 1])
        occ = np.zeros(shape, bool)
        occ[ij[:, 0], ij[:, 1]] = True
        dil = occ.copy()
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                dil |= np.roll(np.roll(occ, dx, 0), dy, 1)
        label = np.full(shape, -1, int)
        count = 0
        for start in zip(*np.nonzero(dil)):
            if label[start] >= 0:
                continue
            stack = [start]
            label[start] = count
            while stack:
                x, y = stack.pop()
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < shape[0] and 0 <= ny < shape[1] and dil[nx, ny] \
                            and label[nx, ny] < 0:
                        label[nx, ny] = count
                        stack.append((nx, ny))
            count += 1
        self.wall_pts = pts
        self.wall_label = label[ij[:, 0], ij[:, 1]]
        self.wall_half = 0.5 * float(np.median(self.w[walls])) if len(walls) else 0.2
        self.n_islands = count
        self.expected_copies = copies

    def _box(self, pts):
        return np.array([pts.min(0) - self.wall_half, pts.max(0) + self.wall_half])

    def _kmeans(self, pts, k, iters=25):
        """Deterministic k-means (farthest-point init) on 2D points."""
        cent = [pts[0]]
        for _ in range(1, k):
            d = np.min([np.linalg.norm(pts - c, axis=1) for c in cent], axis=0)
            cent.append(pts[int(np.argmax(d))])
        cent = np.array(cent)
        for _ in range(iters):
            lab = np.argmin(np.linalg.norm(pts[:, None] - cent[None], axis=2), axis=1)
            new = np.array([pts[lab == c].mean(0) if np.any(lab == c) else cent[c]
                            for c in range(k)])
            if np.allclose(new, cent):
                break
            cent = new
        return lab

    def _place_stl(self, stl_path, scale):
        tris = mg.load_stl(stl_path) * float(scale)
        if len(tris) > STL_FACE_BUDGET:
            tris = _decimate(tris, STL_FACE_BUDGET)
        v, f = mg.weld(tris, tol=1e-3)
        lo, hi = v.min(0), v.max(0)
        v = v - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        ext = (hi - lo)[:2]
        top = self.p1[self.model_rows, 2].max() if self.model_rows.any() else hi[2] - lo[2]
        zfit = top / max(hi[2] - lo[2], 1e-6)
        if abs(zfit - 1) > 0.05:
            self.log(f"gcodeprint: STL height {hi[2] - lo[2]:.2f} mm vs print {top:.2f} mm; "
                     f"rescaling by {zfit:.3f}")
            v *= zfit
            ext = ext * zfit
        self.stl_v, self.stl_f = v, f
        # Slicers may rotate parts by any angle when arranging, so footprint
        # sizes are only bounded: between the STL's short side and diagonal.
        tol = max(1.5, 0.04 * float(ext.max()))
        diag = float(np.linalg.norm(ext))

        def within(box):
            return np.all(box[1] - box[0] <= diag + tol)

        def fits(box):
            size = box[1] - box[0]
            return within(box) and size.max() >= ext.min() - tol

        if self.labelled:
            self._objects_as_copies(ext)
            return
        pts, lab = self.wall_pts, self.wall_label
        boxes = [self._box(pts[lab == k]) for k in range(self.n_islands)]
        # Merge islands that are fragments of one copy (nested rings, posts).
        order = sorted(range(len(boxes)), key=lambda k: -np.prod(boxes[k][1] - boxes[k][0]))
        group_of = np.full(max(len(boxes), 1), -1, int)
        groups = []
        for k in order:
            if group_of[k] >= 0:
                continue
            box = boxes[k].copy()
            group_of[k] = len(groups)
            for j in order:
                if group_of[j] >= 0:
                    continue
                cand = np.array([np.minimum(box[0], boxes[j][0]), np.maximum(box[1], boxes[j][1])])
                if within(cand):
                    group_of[j] = len(groups)
                    box = cand
            groups.append(box)
        clab = group_of[lab] if len(lab) else lab
        want = self.expected_copies or 0
        if want > len(groups) and len(pts) >= want:
            # Copies touching each other: split the footprint with k-means.
            km = self._kmeans(pts, want)
            kboxes = [self._box(pts[km == c]) for c in range(want)]
            if all(fits(b) for b in kboxes):
                groups, clab = kboxes, km
            else:
                self.log(f"gcodeprint: could not split {len(groups)} island group(s) into "
                         f"{want} copies")
        elif want and want != len(groups):
            self.log(f"gcodeprint: found {len(groups)} copies, spec says {want}")
        self.copies = []
        for c, box in enumerate(groups):
            if not fits(box):
                self.log(f"gcodeprint: copy {c} footprint {np.round(box[1] - box[0], 1)} does "
                         f"not fit STL {np.round(ext, 1)}")
            self.copies.append({"centre": (box[0] + box[1]) / 2, "rot": 0.0, "box": box,
                                "shift": np.zeros(2)})
        self._assign_rows()
        for c in range(len(self.copies)):
            self._fit_rotation(c)

    def _objects_as_copies(self, ext):
        """One copy per slicer object label (EXCLUDE_OBJECT / M486 / Bambu
        object markers): exact assignment of rows, box from its walls."""
        objs = self.tl.obj
        ids = [int(o["id"]) for o in self.tl.objects]
        self.copies, self.copy = [], np.full(len(objs), -1, int)
        self.wall_half = 0.2
        for oid in ids:
            mine = objs == oid
            rows = np.flatnonzero(mine & self.ext & np.isin(self.kind, WALL_KINDS))
            if len(rows) == 0:
                rows = np.flatnonzero(mine & self.model_rows)
            if len(rows) == 0:
                continue
            self.wall_half = 0.5 * float(np.median(self.w[rows]))
            pts = self._sample(rows)[:, :2]
            box = self._box(pts)
            c = len(self.copies)
            self.copies.append({"centre": (box[0] + box[1]) / 2, "rot": 0.0, "box": box,
                                "shift": np.zeros(2), "object_id": oid})
            self.copy[mine] = c
        if self.expected_copies and len(self.copies) != self.expected_copies:
            self.log(f"gcodeprint: {len(self.copies)} labelled objects, spec says "
                     f"{self.expected_copies}")
        # Extrusions inside an object that are not walls/infill (gap fill...)
        # belong to it; unlabelled model rows go to the nearest object.
        self.model_rows = self.model_rows | (self.ext & (self.kind == 10) & (self.copy >= 0))
        lost = (self.copy < 0) & (self.model_rows | self.support_rows)
        if lost.any() and self.copies:
            cents = np.array([c["centre"] for c in self.copies])
            mids = 0.5 * (self.p0[lost, :2] + self.p1[lost, :2])
            self.copy[lost] = np.argmin(np.linalg.norm(mids[:, None] - cents[None], axis=2), 1)
        for c in range(len(self.copies)):
            self._fit_rotation(c)

    @property
    def alignment(self):
        """Per-copy STL/toolpath fit (F1, 0..1) and an overall verdict."""
        fits = [c.get("fit", 0.0) for c in self.copies]
        return {"ok": bool(fits) and min(fits) >= ALIGN_MIN_FIT, "fit": fits,
                "labelled": self.labelled}

    def _assign_rows(self):
        """Model/support rows go to the copy whose box (+3 mm) holds their
        midpoint (nearest centre on ties or misses). Other extrusions inside
        a box (gap fill, ironing...) count as model rows; the rest are extras."""
        self.copy = np.full(len(self.p0), -1, int)
        if not self.copies:
            return
        # Only extrusions need a copy; skipping travel rows keeps the
        # (rows x copies) arrays small on multi-million-move timelines.
        ids = np.flatnonzero(self.ext)
        mids = 0.5 * (self.p0[ids, :2] + self.p1[ids, :2])
        cents = np.array([c["centre"] for c in self.copies])
        dist = np.full((len(ids), len(self.copies)), np.inf, np.float32)
        for c, cp in enumerate(self.copies):
            b = cp["box"]
            inside = np.all((mids >= b[0] - 3) & (mids <= b[1] + 3), axis=1)
            dist[inside, c] = np.linalg.norm(mids[inside] - cents[c], axis=1)
        any_in = np.isfinite(dist).any(axis=1)
        self.copy[ids[any_in]] = np.argmin(dist[any_in], axis=1)
        inside_any = np.zeros(len(self.p0), bool)
        inside_any[ids[any_in]] = True
        self.model_rows = self.model_rows | (self.ext & (self.kind == 10) & inside_any)
        lost = ~inside_any & (self.model_rows | self.support_rows)
        if lost.any():
            lm = 0.5 * (self.p0[lost, :2] + self.p1[lost, :2])
            self.copy[lost] = np.argmin(np.linalg.norm(lm[:, None] - cents[None], axis=2), axis=1)

    def _fit_rotation(self, c):
        """Find the Z rotation of the STL that best matches copy c's toolpath:
        side-wall samples of the STL against outer-wall extrusions, as 1 mm
        cells in three height bands, scored by F1 with one cell of slack.
        Quarter turns first, then a 6-degree sweep and 1-degree refinement
        if needed. The rotated STL's footprint box is centred on the copy's."""
        sel = self.model_rows & (self.copy == c)
        rows = np.flatnonzero(sel & (self.kind == 1))
        if len(rows) < 10:
            rows = np.flatnonzero(sel)
        if len(rows) == 0:
            return
        tp = self._sample(rows, step=0.7, cap=15000)
        tp[:, :2] -= self.copies[c]["centre"]
        sp = self._stl_samples()
        zmax = max(tp[:, 2].max(), 1e-3)
        band_t = np.clip((tp[:, 2] / zmax * 3).astype(int), 0, 2)
        band_s = np.clip((sp[:, 2] / zmax * 3).astype(int), 0, 2)
        ring = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)])

        def cells(xy, band, grow=False):
            ij = np.floor(xy).astype(np.int64) + 4000
            if grow:
                ij = (ij[:, None, :] + ring[None]).reshape(-1, 2)
                band = np.repeat(band, len(ring))
            return np.unique(band * 10**8 + ij[:, 0] * 10**4 + ij[:, 1])

        target = cells(tp[:, :2], band_t)
        target_g = cells(tp[:, :2], band_t, grow=True)

        def score(deg):
            a = math.radians(deg)
            r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            q = sp[:, :2] @ r.T
            shift = -(q.min(0) + q.max(0)) / 2
            got = cells(q + shift, band_s)
            got_g = cells(q + shift, band_s, grow=True)
            prec = len(np.intersect1d(got, target_g, assume_unique=True)) / max(len(got), 1)
            rec = len(np.intersect1d(target, got_g, assume_unique=True)) / max(len(target), 1)
            return 2 * prec * rec / max(prec + rec, 1e-9), shift

        best = max(((score(d)[0], d) for d in (0, 90, 180, 270)))
        if best[0] < 0.8:
            best = max([best] + [(score(d)[0], d) for d in range(0, 360, 6)])
            best = max([best] + [(score(best[1] + d)[0], best[1] + d)
                                 for d in (-4, -2, -1, 1, 2, 4)])
        fit, deg = best
        self.copies[c]["rot"] = math.radians(deg)
        self.copies[c]["shift"] = score(deg)[1]
        self.copies[c]["fit"] = round(fit, 3)
        if fit < 0.5:
            self.log(f"gcodeprint: copy {c} STL/toolpath fit only {fit:.2f}")

    def _stl_samples(self, n=15000):
        """Area-weighted samples on the STL's steep (side-wall) faces."""
        if getattr(self, "_stl_pts", None) is None:
            tri = self.stl_v[self.stl_f]
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            area = 0.5 * np.linalg.norm(nrm, axis=1)
            steep = np.abs(nrm[:, 2]) < 0.7 * np.maximum(2 * area, 1e-12)
            if area[steep].sum() > 0:
                area = np.where(steep, area, 0.0)
            rng = np.random.default_rng(0)
            pick = rng.choice(len(tri), size=n, p=area / area.sum())
            u, w = rng.random((2, n))
            flip = u + w > 1
            u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
            t = tri[pick]
            self._stl_pts = (t[:, 0] + (t[:, 1] - t[:, 0]) * u[:, None]
                             + (t[:, 2] - t[:, 0]) * w[:, None])
        return self._stl_pts

    def copy_matrix(self, c):
        """4x4 matrix: STL-local mm -> bed-local metres for copy c."""
        cp = self.copies[c]
        ca, sa = math.cos(cp["rot"]), math.sin(cp["rot"])
        m = np.eye(4)
        m[:3, :3] = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]]) * 1e-3
        m[:3, 3] = self.to_bed([*(cp["centre"] + cp["shift"]), 0.0])
        return m

    # -- supports -------------------------------------------------------------

    def _decimate_supports(self):
        rows = np.flatnonzero(self.support_rows)
        if len(rows) == 0:
            self.sup = {k: np.zeros((0, 3)) if k in ("p0", "p1") else np.zeros(0)
                        for k in ("p0", "p1", "w", "h", "t1", "copy", "layer")}
            return
        p0, p1 = self.p0[rows], self.p1[rows]
        d = p1[:, :2] - p0[:, :2]
        d /= np.maximum(np.linalg.norm(d, axis=1), 1e-9)[:, None]
        cont = np.zeros(len(rows), bool)
        cont[1:] = ((rows[1:] == rows[:-1] + 1)
                    & (np.linalg.norm(p0[1:] - p1[:-1], axis=1) < 1e-4)
                    & (np.einsum("ij,ij->i", d[1:], d[:-1]) > 0.999))
        start = np.flatnonzero(~cont)
        end = np.r_[start[1:] - 1, len(rows) - 1]
        layer = self.tl.layer[rows[start]]
        sup = {"p0": p0[start], "p1": p1[end], "w": self.w[rows[start]],
               "h": self.h[rows[start]].copy(), "t1": self.t1[rows[end]],
               "copy": self.copy[rows[start]], "layer": layer}
        k = int(math.ceil(len(start) / SUPPORT_BUDGET))
        if k > 1:
            keep = layer % k == 0
            sup = {key: val[keep] for key, val in sup.items()}
            sup["h"] = sup["h"] * k
        self.sup = sup

    # -- queries ------------------------------------------------------------

    def completed(self, t):
        return self.t1 <= t

    def top_z(self, t, c):
        """Height (mm) of the highest completed or partial model row of copy c."""
        rows = self.model_rows & (self.copy == c) & (self.t0 < t)
        return float(self.p1[rows, 2].max()) if rows.any() else 0.0

    def cut_z(self, top):
        """Top of the layer N_TOP layers below `top` (mm), 0 if none."""
        i = int(np.searchsorted(self.levels, top + 1e-4, side="right")) - 1
        j = i - N_TOP
        return float(self.levels[j]) if j >= 0 else 0.0

    def level_below(self, z):
        i = int(np.searchsorted(self.levels, z - 1e-4, side="left")) - 1
        return float(self.levels[i]) if i >= 0 else 0.0

    def rows_done(self, t, mask):
        """Row ids of `mask` completed by t, plus (id, fraction) of the
        partial move when it belongs to `mask`."""
        done = np.flatnonzero(mask & (self.t1 <= t))
        part = self.tl.partial(t)
        if part is not None and mask[part[0]]:
            return done, part
        return done, None

    def body(self, c, z0, z1):
        """STL of copy c between z0 and z1 (mm), in bed-local metres:
        (verts, side faces, cap faces)."""
        if z1 <= z0 + 1e-6:
            return np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 3), int)
        key = (round(z0, 4), round(z1, 4))
        if key not in self._slab_cache:
            if len(self._slab_cache) > 16:
                self._slab_cache.clear()
            v, (sides, caps) = mg.slab(self.stl_v, self.stl_f, z0, z1)
            self._slab_cache[key] = (v, sides, caps)
        v, sides, caps = self._slab_cache[key]
        return mg.transform(v, self.copy_matrix(c)), sides, caps

    def beads(self, rows, partial=None, perturb=None):
        """Bead geometry for timeline rows (plus a partial move). `perturb`,
        if given, is called as perturb(row_ids, p0, p1, w, h) in mm and
        returns new (p0, p1, w, h); it must be a pure function of row ids so
        a row looks the same in every keyframe."""
        ids = np.asarray(rows, int)
        p0, p1 = self.p0[ids], self.p1[ids]
        w, h = self.w[ids], self.h[ids]
        if partial is not None:
            i, f = partial
            q = self.p0[i] + (self.p1[i] - self.p0[i]) * f
            ids = np.r_[ids, i]
            p0 = np.concatenate([p0, self.p0[i:i + 1]])
            p1 = np.concatenate([p1, q[None]])
            w = np.r_[w, self.w[i]]
            h = np.r_[h, self.h[i]]
        if len(p0) > TOP_BUDGET:
            keep = np.linspace(0, len(p0) - 1, TOP_BUDGET).astype(int)
            ids, p0, p1, w, h = ids[keep], p0[keep], p1[keep], w[keep], h[keep]
        if perturb is not None:
            p0, p1, w, h = perturb(ids, p0, p1, w, h)
        return mg.beads(self.to_bed(p0), self.to_bed(p1), w * 1e-3, h * 1e-3)

    def support_beads(self, t, zmax, copies=None, zmin=-1.0):
        s = self.sup
        sel = (s["t1"] <= t) & (s["p1"][:, 2] <= zmax + 1e-4) & (s["p1"][:, 2] > zmin + 1e-4)
        if copies is not None:
            sel &= np.isin(s["copy"], copies)
        return mg.beads(self.to_bed(s["p0"][sel]), self.to_bed(s["p1"][sel]),
                        s["w"][sel] * 1e-3, s["h"][sel] * 1e-3)

    def extras(self, t, plate_half):
        """Brim, skirt, purge and stray extrusions outside the copies."""
        mask = self.ext & ~self.model_rows & ~self.support_rows
        rows, part = self.rows_done(t, mask)
        p0, p1 = self.to_bed(self.p0[rows]), self.to_bed(self.p1[rows])
        # Purge lines just off the printable area still land on the sheet.
        lim = np.array([plate_half[0] - 0.0015, plate_half[1] - 0.0015, 1.0])
        ok = (np.abs(p0[:, :2]) < np.array(plate_half) + 0.01).all(1) & \
             (np.abs(p1[:, :2]) < np.array(plate_half) + 0.01).all(1) & (p1[:, 2] < 0.002)
        p0 = np.clip(p0[ok], -lim, lim)
        p1 = np.clip(p1[ok], -lim, lim)
        return mg.beads(p0, p1, self.w[rows][ok] * 1e-3, self.h[rows][ok] * 1e-3)

    def footprint(self, c, margin=0.0):
        """Copy c's XY box in bed-local metres (x0, y0, x1, y1)."""
        box = self.copies[c]["box"]
        lo = self.to_bed([*box[0], 0])[:2] - margin
        hi = self.to_bed([*box[1], 0])[:2] + margin
        return np.r_[lo, hi]


def _decimate(tris, target):
    """Vertex-clustering decimation of a triangle soup to roughly `target`."""
    pts = tris.reshape(-1, 3)
    size = pts.max(0) - pts.min(0)
    cell = (np.prod(size) / (target * 0.5)) ** (1 / 3) if np.prod(size) > 0 else 0.1
    for _ in range(6):
        key = np.floor(pts / cell).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        f = inv.reshape(-1, 3)
        good = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        if good.sum() <= target:
            break
        cell *= 1.3
    sums = np.zeros((inv.max() + 1, 3))
    np.add.at(sums, inv, pts)
    cnt = np.bincount(inv)
    cent = sums / cnt[:, None]
    return cent[f[good]]


# ---------------------------------------------------------------------------
# Scene side
# ---------------------------------------------------------------------------

class PrintObjects:
    """A pool of mesh objects keyed by (name), each with one material and a
    pass index, parented to the bed. `update` rewrites all of them."""

    def __init__(self, coll, parent):
        self.coll, self.parent = coll, parent
        self.objs = {}

    def get(self, name, material, pass_index, coll=None):
        obj = self.objs.get(name)
        if obj is None:
            mesh = bpy.data.meshes.new(name)
            obj = bpy.data.objects.new(name, mesh)
            (coll or self.coll).objects.link(obj)
            obj.parent = self.parent
            mesh.materials.append(material)
            obj.pass_index = pass_index
            self.objs[name] = obj
        return obj

    def update(self, pieces):
        """`pieces`: list of (name, material, pass_index, (verts, faces)).
        Pieces sharing a name are merged; objects not mentioned are emptied."""
        merged = {}
        for name, mat, idx, geom in pieces:
            entry = merged.setdefault(name, [mat, idx, []])
            entry[2].append(geom)
        stats = {}
        for name, (mat, idx, geoms) in merged.items():
            obj = self.get(name, mat, idx)
            if obj.pass_index != idx:
                obj.pass_index = idx
            v, f = mg.merge(geoms)
            stats[name] = mg.write_mesh(obj.data, v, f)
            obj.hide_render = stats[name] == 0
        for name, obj in self.objs.items():
            if name not in merged:
                mg.write_mesh(obj.data, np.zeros((0, 3)), [])
                obj.hide_render = True
        return stats


def _xf(geom, matrix=None, offset=None):
    v, f = geom
    if matrix is not None:
        v = mg.transform(v, matrix)
    if offset is not None and len(v):
        v = v + offset
    return v, f


def part_pieces(plan, c, t, m, *, freeze=None, zr=(-1.0, 1e9), rows_tr=None, rows_zr=None,
                matrix=None, offset=None, pass_index=PASS_PRINT, prefix="Print", perturb=None,
                body_t=None):
    """Pieces for copy c at time t, optionally restricted and moved.

    freeze   the part stops growing at this time (later rows are not added)
    zr       (z0, z1] in mm: the STL body slab and supports to include
    rows_zr  same for toolpath rows (defaults to zr); rows_tr = [t0, t1)
             additionally selects rows by their start time
    body_t   time whose top layer sets the STL cut (default: growth time)
    matrix   4x4 bed-local transform; offset a bed-local translation (m)
    """
    tg = t if freeze is None else min(t, freeze)
    tb = tg if body_t is None else min(tg, body_t)
    top = plan.top_z(tb, c)
    cut = plan.cut_z(top) if top > 0 else 0.0
    z0, z1 = zr
    rz0, rz1 = rows_zr or zr
    pieces = []
    bz0, bz1 = max(z0, 0.0), min(z1, cut)
    if bz1 > bz0 + 1e-6:
        v, sides, caps = plan.body(c, bz0, bz1)
        v = _xf((v, []), matrix, offset)[0]
        pieces.append((f"{prefix}Body", m["pla"], pass_index, (v, [sides])))
        pieces.append((f"{prefix}Cap", m["cap"], pass_index, (v, [caps])))
    z = plan.p1[:, 2]
    mask = ((plan.model_rows | plan.support_rows) & (plan.copy == c)
            & (z > max(cut, rz0) + 1e-4) & (z <= rz1 + 1e-4))
    if rows_tr is not None:
        mask &= (plan.t0 >= rows_tr[0]) & (plan.t0 < rows_tr[1])
    rows, part = plan.rows_done(tg, mask)
    pieces.append((f"{prefix}Top", m["pla_top"], pass_index,
                   _xf(plan.beads(rows, part, perturb), matrix, offset)))
    pieces.append((f"{prefix}Supports", m["pla"], pass_index,
                   _xf(plan.support_beads(tg, min(cut, z1), [c], zmin=z0), matrix, offset)))
    return pieces


def normal_pieces(plan, t, m, plate_half, copies=None):
    """Geometry of an undisturbed print at time t."""
    pieces = []
    for c in range(len(plan.copies)) if copies is None else copies:
        pieces += part_pieces(plan, c, t, m)
    pieces.append(("Extras", m["pla_top"], PASS_PRINT, plan.extras(t, plate_half)))
    return pieces
