"""NumPy mesh builders for geometry that changes every keyframe.

Everything returns plain arrays: `verts` (N, 3) float and `faces`, a list of
(F, k) int arrays (triangles, quads, hexagons...). `write_mesh` loads them into
an existing mesh datablock with `foreach_set`, which is orders of magnitude
faster than bmesh or curve bevels for 1e5-1e6 faces.
"""

import struct

import numpy as np
from mathutils import Vector
from mathutils.geometry import tessellate_polygon

# Bead cross-section (offset across the line in widths, height in layer
# heights above the bead centre): flat-ish top and bottom, rounded sides.
_BEAD = np.array([(-0.5, 0.0), (-0.3, 0.5), (0.3, 0.5), (0.5, 0.0), (0.3, -0.5),
                  (-0.3, -0.5)])


def empty_geometry():
    return np.zeros((0, 3)), []


def merge(parts):
    """Concatenate several (verts, faces) results."""
    verts, faces, base = [], [], 0
    for v, fs in parts:
        if len(v) == 0:
            continue
        verts.append(v)
        faces.extend(f + base for f in fs if len(f))
        base += len(v)
    if not verts:
        return empty_geometry()
    return np.concatenate(verts), faces


def write_mesh(mesh, verts, faces, smooth=True):
    """Replace the geometry of `mesh` with `verts`/`faces`."""
    mesh.clear_geometry()
    faces = [f for f in faces if len(f)]
    if len(verts) == 0 or not faces:
        mesh.update()
        return 0
    sizes = np.concatenate([np.full(len(f), f.shape[1]) for f in faces])
    loops = np.concatenate([f.ravel() for f in faces]).astype(np.int32)
    mesh.vertices.add(len(verts))
    mesh.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    mesh.loops.add(len(loops))
    mesh.loops.foreach_set("vertex_index", loops)
    mesh.polygons.add(len(sizes))
    starts = np.zeros(len(sizes), np.int32)
    np.cumsum(sizes[:-1], out=starts[1:])
    mesh.polygons.foreach_set("loop_start", starts)
    mesh.polygons.foreach_set("use_smooth", np.full(len(sizes), smooth))
    mesh.update(calc_edges=True)
    return len(sizes)


def transform(verts, matrix):
    """Apply a 4x4 matrix (numpy) to (N, 3) points."""
    if matrix is None or len(verts) == 0:
        return verts
    return verts @ matrix[:3, :3].T + matrix[:3, 3]


# ---------------------------------------------------------------------------
# Extrusion beads
# ---------------------------------------------------------------------------

def beads(p0, p1, width, height, extend=0.35, caps=True):
    """One bead per segment: a 6-sided flattened tube from p0 to p1 whose top
    is at the segment Z (the nozzle height) and whose bottom is `height` lower.
    Ends are pushed out by `extend` widths so consecutive lines join up."""
    d = p1[:, :2] - p0[:, :2]
    length = np.linalg.norm(d, axis=1)
    ok = length > 1e-7
    p0, p1, width, height, d, length = (a[ok] for a in (p0, p1, width, height, d, length))
    n = len(p0)
    if n == 0:
        return empty_geometry()
    d = d / length[:, None]
    ext = np.minimum(extend * width, 0.5 * length + 0.5 * width)
    a = p0.copy()
    b = p1.copy()
    a[:, :2] -= d * ext[:, None]
    b[:, :2] += d * ext[:, None]
    side = np.stack([-d[:, 1], d[:, 0]], axis=1)
    ring = np.empty((n, 2, 6, 3))
    for e, end in enumerate((a, b)):
        zc = end[:, 2] - 0.5 * height
        for k, (s, z) in enumerate(_BEAD):
            ring[:, e, k, :2] = end[:, :2] + side * (s * width)[:, None]
            ring[:, e, k, 2] = zc + z * height
    verts = ring.reshape(-1, 3)
    base = (np.arange(n) * 12)[:, None]
    k = np.arange(6)
    k2 = (k + 1) % 6
    quads = np.stack([k, k2, 6 + k2, 6 + k], axis=1)  # (6, 4)
    faces = [(base[:, None, :] + quads[None]).reshape(-1, 4)]
    if caps:
        faces.append(base + k[::-1][None])
        faces.append(base + 6 + k[None])
    return verts, faces


# ---------------------------------------------------------------------------
# Tubes along polylines (filament strands)
# ---------------------------------------------------------------------------

def tubes(points, breaks, radius, sides=5, squash=1.0):
    """Tubes through `points` (N, 3); `breaks` is a bool array, True where a
    new strand starts. Radius may be a scalar or per point."""
    n = len(points)
    if n < 2:
        return empty_geometry()
    breaks = breaks.copy()
    breaks[0] = True
    sid = np.cumsum(breaks) - 1
    fwd = np.zeros_like(points)
    nxt_same = np.zeros(n, bool)
    nxt_same[:-1] = sid[1:] == sid[:-1]
    prv_same = np.zeros(n, bool)
    prv_same[1:] = nxt_same[:-1]
    fwd[:-1][nxt_same[:-1]] = (points[1:] - points[:-1])[nxt_same[:-1]]
    bwd = np.zeros_like(points)
    bwd[1:][prv_same[1:]] = (points[1:] - points[:-1])[prv_same[1:]]
    tang = fwd + bwd
    tl = np.linalg.norm(tang, axis=1)
    tang = np.where(tl[:, None] > 1e-12, tang / np.maximum(tl, 1e-12)[:, None],
                    np.array([1.0, 0, 0]))
    up = np.where(np.abs(tang[:, 2:3]) > 0.9, np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))
    u = np.cross(tang, up)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(tang, u)
    r = np.broadcast_to(np.asarray(radius, float), (n,))
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = (points[:, None, :]
            + (np.cos(ang)[None, :, None] * u[:, None, :]
               + np.sin(ang)[None, :, None] * v[:, None, :] * np.array([1, 1, squash]))
            * r[:, None, None])
    verts = ring.reshape(-1, 3)
    seg = np.flatnonzero(nxt_same)
    k = np.arange(sides)
    k2 = (k + 1) % sides
    base = (seg * sides)[:, None, None]
    quads = np.stack([k, k2, sides + k2, sides + k], axis=1)[None]
    faces = [(base + quads).reshape(-1, 4)]
    starts = np.flatnonzero(breaks)
    ends = np.r_[starts[1:] - 1, n - 1]
    faces.append((starts * sides)[:, None] + k[::-1][None])
    faces.append((ends * sides)[:, None] + k[None])
    return verts, faces


def blob(center, size, seed, detail=3, lumps=0.35, noise_scale=2.0):
    """Lumpy ellipsoid (icosphere-like via a subdivided octahedron)."""
    from mathutils import noise as mnoise
    verts = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
                     float)
    tris = np.array([(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5),
                     (3, 1, 5), (0, 3, 5)])
    for _ in range(detail + 1):
        edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]),
                        axis=1)
        uniq, inv = np.unique(edges, axis=0, return_inverse=True)
        mids = verts[uniq].mean(axis=1)
        mids /= np.linalg.norm(mids, axis=1)[:, None]
        mid = len(verts) + inv.reshape(3, -1).T
        verts = np.concatenate([verts, mids])
        a, b, c = tris.T
        ab, bc, ca = mid.T
        tris = np.concatenate([np.stack(x, 1) for x in
                               ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    off = Vector((seed % 97 * 0.37, seed % 89 * 0.53, seed % 83 * 0.71))
    scale = np.array([1 + lumps * mnoise.noise(Vector(p) * noise_scale + off) for p in verts])
    verts = verts * scale[:, None] * np.asarray(size) + np.asarray(center)
    return verts, [tris]


# ---------------------------------------------------------------------------
# STL loading and slab clipping with caps
# ---------------------------------------------------------------------------

def load_stl(path):
    """Triangle soup (T, 3, 3) in file units, binary or ASCII."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * count == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                                      ("a", "<u2")]), count=count, offset=84)
            return rec["v"].astype(np.float64)
    vals = [line.split()[1:4] for line in data.decode("ascii", "replace").splitlines()
            if line.strip().startswith("vertex")]
    return np.array(vals, float).reshape(-1, 3, 3)


def weld(tris, tol=1e-4):
    """Indexed mesh from a triangle soup, merging vertices closer than `tol`."""
    pts = tris.reshape(-1, 3)
    key = np.round(pts / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    faces = inv.reshape(-1, 3)
    good = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return pts[first], faces[good]


def _clip(verts, faces, z, keep_below):
    """Clip a triangle mesh by the plane Z = z. Returns (verts, faces, cut
    segments as pairs of vertex ids on the plane)."""
    s = verts[:, 2] - z
    if not keep_below:
        s = -s
    eps = 1e-9
    s = np.where(np.abs(s) < eps, eps, s)  # nothing exactly on the plane
    inside = s < 0
    fin = inside[faces]
    cnt = fin.sum(1)
    keep = faces[cnt == 3]
    part = faces[(cnt == 1) | (cnt == 2)]
    if len(part) == 0:
        return verts, keep, np.zeros((0, 2), int)
    pin = inside[part]
    # Rotate each triangle so its odd-one-out vertex comes first.
    one = (pin.sum(1) == 1)
    odd = np.where(one, np.argmax(pin, axis=1), np.argmax(~pin, axis=1))
    idx = (odd[:, None] + np.arange(3)[None]) % 3
    tri = np.take_along_axis(part, idx, axis=1)
    a, b, c = tri.T
    # New points on edges a-b and a-c, shared between neighbours by edge key.
    e = np.concatenate([np.sort(np.stack([a, b], 1), 1), np.sort(np.stack([a, c], 1), 1)])
    uniq, inv = np.unique(e, axis=0, return_inverse=True)
    i, j = uniq.T
    tpar = s[i] / (s[i] - s[j])
    newv = verts[i] + (verts[j] - verts[i]) * tpar[:, None]
    newv[:, 2] = z
    base = len(verts)
    m = len(tri)
    ab = base + inv[:m]
    ac = base + inv[m:]
    out = [keep]
    o1 = one
    # Odd vertex inside: keep the tip triangle (a, ab, ac).
    out.append(np.stack([a[o1], ab[o1], ac[o1]], 1))
    # Odd vertex outside: keep the quad (ab, b, c, ac) as two triangles.
    o2 = ~one
    out.append(np.stack([ab[o2], b[o2], c[o2]], 1))
    out.append(np.stack([ab[o2], c[o2], ac[o2]], 1))
    segs = np.stack([ab, ac], 1)
    return np.concatenate([verts, newv]), np.concatenate(out), segs


def _loops(segs):
    """Chain undirected segments into closed loops of vertex ids."""
    adj = {}
    for u, v in segs:
        if u == v:
            continue
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    seen, loops = set(), []
    for start in adj:
        if start in seen or len(adj[start]) != 2:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nbrs = adj[cur]
            nxt = nbrs[0] if nbrs[0] != prev else nbrs[1] if len(nbrs) > 1 else None
            if nxt is None or len(adj[nxt]) != 2:
                loop = None
                break
            if nxt == start:
                break
            if nxt in seen:
                loop = None
                break
            seen.add(nxt)
            loop.append(nxt)
            prev, cur = cur, nxt
        if loop and len(loop) >= 3:
            loops.append(loop)
    return loops


def _cap(verts, segs, up):
    loops = _loops(segs)
    if not loops:
        return np.zeros((0, 3), int)
    polys = [[Vector((verts[i, 0], verts[i, 1], 0.0)) for i in lp] for lp in loops]
    try:
        tris = tessellate_polygon(polys)
    except Exception:
        return np.zeros((0, 3), int)
    if not tris:
        return np.zeros((0, 3), int)
    flat = np.concatenate([np.array(lp) for lp in loops])
    tris = flat[np.array(tris)]
    p = verts[tris]
    nz = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
          - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    flip = (nz < 0) if up else (nz > 0)
    tris[flip] = tris[flip][:, ::-1]
    return tris


def slab(verts, faces, z0, z1):
    """The part of a closed mesh between Z = z0 and Z = z1, capped on both
    planes. Returns (verts, [side tris, cap tris]); caps are listed separately
    so they can get their own material."""
    caps = []
    if z1 < verts[:, 2].max():
        verts, faces, segs = _clip(verts, faces, z1, keep_below=True)
        caps.append(_cap(verts, segs, up=True))
    if z0 > verts[:, 2].min():
        # Clipping only appends vertices, so the top cap's ids stay valid.
        verts, faces, segs = _clip(verts, faces, z0, keep_below=False)
        caps.append(_cap(verts, segs, up=False))
    cap = np.concatenate(caps) if caps else np.zeros((0, 3), int)
    return verts, [faces, cap]
