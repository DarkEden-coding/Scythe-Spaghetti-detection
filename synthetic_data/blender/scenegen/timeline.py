"""NumPy-only reader for CONTRACTS.md §3; coordinates are printer mm, time seconds.

``load_timeline(path)`` returns a Timeline exposing every NPZ array as an
attribute, plus ``meta`` as a dict. Queries clamp outside the stored interval;
exact boundaries select the following move, and the final endpoint selects the
last move at fraction 1. ``partial`` returns only an unfinished, positive-E,
nonzero-length extrusion (None for travels/retractions and completed moves).
``extruded_mask`` includes every positive-E row whose end is <= t.
``layer_range`` returns the smallest contiguous slice containing a layer (empty
slice for absent layers); sequential-object G-code can interleave layer indices,
so use ``segments_for_layers`` for exact filtering. That method returns int64
row indices for inclusive l0..l1, optionally filtered by kind codes.
``velocity_at`` is the piecewise constant derivative of the contract's linear
position interpolation, zero outside the timeline. ``state_at`` clamps elapsed
and progress to [0,total_time], and reports print-layer Z rather than hop Z.

The producer keeps explicitly marked startup purge at negative times, drops
other startup motion, and sets t=0 to the first print extrusion. No synthpipe
imports or pickle loading are needed in Blender.
"""

import json

import numpy as np


class Timeline:
    def __init__(self, arrays):
        for key, value in arrays.items():
            if key != "meta":
                setattr(self, key, value)
        raw = arrays["meta"]
        self.meta = raw if isinstance(raw, dict) else json.loads(str(raw.item()))
        self.objects = self.meta.get("objects", [])
        if not hasattr(self, "obj"):
            self.obj = np.full(len(self.t0), -1, dtype=np.int16)
        n = len(self.t0)
        if n == 0 or self.p0.shape != (n, 3) or self.p1.shape != (n, 3):
            raise ValueError("Timeline must contain nonempty (N,3) positions")
        if np.any(self.t1 <= self.t0) or np.any(self.t0[1:] < self.t1[:-1] - 1e-9):
            raise ValueError("Timeline intervals must be positive and ordered")

    def _at(self, t):
        t = float(t)
        if not np.isfinite(t):
            raise ValueError("Time must be finite")
        i = int(np.clip(np.searchsorted(self.t1, t, side="right"), 0, len(self.t1) - 1))
        f = float(np.clip((t - self.t0[i]) / (self.t1[i] - self.t0[i]), 0, 1))
        return i, f

    def nozzle_at(self, t):
        i, f = self._at(t)
        return self.p0[i].astype(np.float64) + (
            self.p1[i].astype(np.float64) - self.p0[i]
        ) * f, i

    def velocity_at(self, t):
        i, _ = self._at(t)
        if t < self.t0[0] or t >= self.t1[-1]:
            return np.zeros(3)
        return (self.p1[i].astype(np.float64) - self.p0[i]) / (self.t1[i] - self.t0[i])

    def state_at(self, t):
        i, _ = self._at(t)
        layer = int(self.layer[i])
        elapsed = float(np.clip(t, 0, self.meta["total_time_s"]))
        z = (
            float(self.layer_z[layer])
            if 0 <= layer < len(self.layer_z)
            else float(self.nozzle_at(t)[0][2])
        )
        return {
            "z_mm": z,
            "layer": layer,
            "progress": elapsed / max(self.meta["total_time_s"], 1e-12),
            "elapsed_s": elapsed,
        }

    def extruded_mask(self, t):
        self._at(t)
        return (self.t1 <= t) & (self.e > 0)

    def partial(self, t):
        i, f = self._at(t)
        if (
            self.t0[i] <= t < self.t1[i]
            and self.e[i] > 0
            and np.any(self.p0[i] != self.p1[i])
        ):
            return i, f
        return None

    def layer_range(self, layer):
        indices = np.flatnonzero(self.layer == layer)
        return (
            slice(int(indices[0]), int(indices[-1]) + 1) if len(indices) else slice(0, 0)
        )

    def object_mask(self, obj_id):
        """Boolean mask of all moves assigned to a normalized object id."""
        return self.obj == obj_id

    def segments_for_object(self, obj_id, kinds=None, t=None):
        """Row indices for an object, optionally by kind and completed at t."""
        mask = self.object_mask(obj_id)
        if kinds is not None:
            mask &= np.isin(self.kind, list(kinds))
        if t is not None:
            self._at(t)  # same finite-time validation as other time queries
            mask &= self.t1 <= t
        return np.flatnonzero(mask)

    def segments_for_layers(self, l0, l1, kinds=None):
        mask = (self.layer >= l0) & (self.layer <= l1)
        if kinds is not None:
            mask &= np.isin(self.kind, list(kinds))
        return np.flatnonzero(mask)


def load_timeline(path):
    with np.load(path, allow_pickle=False) as archive:
        return Timeline({key: archive[key] for key in archive.files})
