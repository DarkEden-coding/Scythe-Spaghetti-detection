"""Render-side events (CONTRACTS.md §4 `events`) and per-sequence lighting.

    lights_off      the setup's main light (chamber LED, ceiling lamp, LED
                    strip) is off during [t0, t1); room spill and the IR
                    illuminator stay. Some draws also make it night outside.
    lights_on       the opposite: the sequence starts dark and the main
                    light is on during [t0, t1)
    daylight_shift  daylight / room light and ambient ramp to a new level
                    and colour over [t0, t1] and stay there
    hand            an arm reaches in towards the plate and back out over
                    [t0, t1] (class 8); enclosures get their door opened
    door_open       door or front panel swung open during [t0, t1) (class 8)
    part_removed    the print is taken off at t0, the plate stays empty with
                    the toolhead parked until t1, then the same print starts
                    again (print time t - t1)

Post-side events (ae_jump, wb_shift, bump) are left to the ISP and only
reported in `events_active`. Every random choice comes from the event's
`params.seed`.
"""

import math
import random

import bpy
import numpy as np
from mathutils import Matrix, Vector

from .materials import principled
from .util import PASS_ENVIRONMENT, PASS_OCCLUDER, box, cylinder, empty

# Lights that `lights_off` switches off, and lights / emitters that make up
# daylight, per setup (object or material names from setups.py).
MAIN = {"enclosed_corexy": ["ChamberLED"], "prusa_desk": ["CeilingLamp"],
        "voron_topcam": ["LEDStripLight", "mat:LEDStrip"], "lack_ir_night": []}
DAYLIGHT = {"enclosed_corexy": ["RoomLamp"], "prusa_desk": ["Daylight", "mat:WindowSky"],
            "voron_topcam": ["RoomLight"], "lack_ir_night": []}
# Never scaled: the camera's own illuminator and the toolhead's LEDs.
FIXED = ("IRIlluminator", "NozzleLight")
# Hinged doors: (object names, hinge (x, y), +1 if the door extends towards +x).
DOORS = {"enclosed_corexy": [(("DoorGlass", "DoorFrameL", "DoorFrameR"), (-0.19, -0.191), 1)],
         "voron_topcam": [(("Door-1", "DoorHandle-1"), (-0.24, -0.2525), 1),
                          (("Door+1", "DoorHandle+1"), (0.24, -0.2525), -1)],
         "lack_ir_night": [(("PanelFront",), (-0.275, -0.277), 1)]}
RENDER_SIDE = {"lights_off", "lights_on", "daylight_shift", "hand", "door_open",
               "part_removed"}
SKIN = [(0.80, 0.55, 0.42), (0.62, 0.40, 0.28), (0.42, 0.26, 0.17), (0.25, 0.15, 0.10),
        (0.88, 0.66, 0.54)]


def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class Events:
    def __init__(self, events, ctx):
        self.ctx = ctx
        self.setup = ctx["setup_name"]
        self.events = sorted(events, key=lambda e: (e["t0"], e["type"]))
        self.scale = ctx.get("light_scale", 1.0)
        self.rngs = [random.Random(int((e.get("params") or {}).get("seed", i)))
                     for i, e in enumerate(self.events)]
        self._register_lights()
        self.draw = []
        for e, rng in zip(self.events, self.rngs):
            d = {}
            if e["type"] == "lights_off":
                d["night"] = self.setup == "prusa_desk" and rng.random() < 0.5
            elif e["type"] == "daylight_shift":
                d["factor"] = rng.choice((rng.uniform(0.15, 0.6), rng.uniform(1.4, 2.5)))
                d["tint"] = rng.choice(((1.0, 0.72, 0.45), (0.75, 0.85, 1.0), (1, 1, 1)))
            elif e["type"] == "door_open":
                d["angle"] = math.radians(rng.uniform(45, 110))
            elif e["type"] == "hand":
                d["angle"] = math.radians(rng.uniform(55, 100))
            self.draw.append(d)
        self.hand = self._build_hand() if any(e["type"] == "hand" for e in self.events) else None
        self.doors = self._build_doors()
        if any(e["type"] == "door_open" for e in self.events) and not self.doors:
            print(f"events: setup {self.setup} has no door; door_open ignored")

    # -- lights ---------------------------------------------------------------

    def _register_lights(self):
        self.lights = []
        for obj in bpy.data.objects:
            if obj.type == "LIGHT":
                self.lights.append((obj, obj.data.energy, tuple(obj.data.color)))
        self.emitters = {}
        for mat in bpy.data.materials:
            bsdf = mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
            if bsdf and bsdf.inputs["Emission Strength"].default_value > 0:
                self.emitters[mat.name] = (bsdf, bsdf.inputs["Emission Strength"].default_value,
                                           tuple(bsdf.inputs["Emission Color"].default_value))
        world = bpy.context.scene.world
        bg = world.node_tree.nodes["Background"]
        self.world = (bg, bg.inputs["Strength"].default_value,
                      tuple(bg.inputs["Color"].default_value))

    def _role(self, name, table):
        return name in table.get(self.setup, [])

    def _lighting(self, t):
        main, day, tint, world = 1.0, 1.0, np.ones(3), 1.0
        starts_dark = any(e["type"] == "lights_on" for e in self.events)
        if starts_dark:
            main = 0.0
        for e, d in zip(self.events, self.draw):
            inside = e["t0"] <= t < e["t1"]
            if e["type"] == "lights_off" and inside:
                main = 0.0
                if d["night"]:
                    day, world = 0.04, 0.15
            elif e["type"] == "lights_on" and inside:
                main = 1.0
            elif e["type"] == "daylight_shift" and t >= e["t0"]:
                s = _smooth((t - e["t0"]) / max(e["t1"] - e["t0"], 1e-6))
                day *= 1 + (d["factor"] - 1) * s
                world *= 1 + (d["factor"] - 1) * s * 0.7
                tint = tint * (1 - s) + np.array(d["tint"]) * s
        return main, day, tint, world

    def _apply_lights(self, t):
        main, day, tint, world = self._lighting(t)
        for obj, energy, color in self.lights:
            if obj.name.startswith(FIXED):
                continue
            k = self.scale
            col = np.array(color)
            if self._role(obj.name, MAIN):
                k *= main
            elif self._role(obj.name, DAYLIGHT):
                k *= day
                col = col * tint
            obj.data.energy = energy * k
            obj.data.color = tuple(col)
        for name, (bsdf, strength, color) in self.emitters.items():
            k, col = 1.0, np.array(color[:3])
            if self._role("mat:" + name, MAIN):
                k = main
            elif self._role("mat:" + name, DAYLIGHT):
                k, col = day, col * tint
            else:
                continue
            bsdf.inputs["Emission Strength"].default_value = strength * k * self.scale
            bsdf.inputs["Emission Color"].default_value = (*col, 1.0)
        bg, strength, color = self.world
        bg.inputs["Strength"].default_value = strength * world * self.scale
        bg.inputs["Color"].default_value = (*(np.array(color[:3]) * tint), 1.0)

    # -- doors ----------------------------------------------------------------

    def _build_doors(self):
        doors = []
        for names, hinge, direction in DOORS.get(self.setup, []):
            objs = [bpy.data.objects[n] for n in names if n in bpy.data.objects]
            if not objs:
                continue
            pivot = empty(f"DoorHinge{len(doors)}", self.ctx["colls"]["Environment"],
                          (hinge[0], hinge[1], 0.0))
            for obj in objs:
                mw = obj.matrix_world.copy()
                obj.parent = pivot
                obj.matrix_parent_inverse = pivot.matrix_world.inverted()
                obj.matrix_world = mw
            doors.append((pivot, objs, direction))
        return doors

    def _apply_doors(self, t):
        angle = 0.0
        for e, d in zip(self.events, self.draw):
            if e["type"] == "door_open" and e["t0"] <= t < e["t1"]:
                angle = max(angle, d["angle"])
            if e["type"] == "hand" and e["t0"] <= t < e["t1"]:
                angle = max(angle, d["angle"])
        for pivot, objs, direction in self.doors:
            pivot.rotation_euler = (0, 0, -direction * angle)
            for obj in objs:
                obj.pass_index = PASS_OCCLUDER if angle > math.radians(4) else PASS_ENVIRONMENT
        return angle > 0

    # -- hand -----------------------------------------------------------------

    def _build_hand(self):
        """Forearm + palm + fingers along local +X, fingertips at the origin,
        palm facing -Z."""
        rng = random.Random(self.rngs[[e["type"] for e in self.events].index("hand")].random())
        c = self.ctx["colls"]["Environment"]
        skin = principled("Skin", rng.choice(SKIN), 0.5,
                          **{"Subsurface Weight": 0.3, "Subsurface Radius": (1.0, 0.4, 0.25),
                             "Subsurface Scale": 0.004})
        sleeve = principled("Sleeve", rng.choice([(0.05, 0.05, 0.06), (0.15, 0.2, 0.35),
                                                  (0.4, 0.05, 0.05), (0.5, 0.5, 0.48)]), 0.8)
        root = empty("Hand", c)
        s = rng.uniform(0.9, 1.1)
        P = PASS_OCCLUDER
        box("Palm", (0.115 * s, 0, 0), (0.09 * s, 0.085 * s, 0.026 * s), skin, c, P, bevel=0.01,
            parent=root)
        cylinder("Forearm", (0.30 * s, 0, 0.004), 0.031 * s, 0.26 * s, skin, c, axis="X",
                 verts=20, pass_index=P, parent=root, cone_r2=0.026 * s, bevel=0.004)
        if rng.random() < 0.7:
            cylinder("SleeveCuff", (0.40 * s, 0, 0.004), 0.042 * s, 0.14 * s, sleeve, c,
                     axis="X", verts=20, pass_index=P, parent=root, bevel=0.006)
        curl = rng.uniform(0.0, 0.5)
        for i, y in enumerate((-0.03, -0.01, 0.01, 0.03)):
            length = (0.07, 0.08, 0.075, 0.06)[i] * s
            f = cylinder(f"Finger{i}", (0.07 * s - length / 2, y * s, -0.003), 0.0085 * s,
                         length, skin, c, axis="X", verts=10, pass_index=P, parent=root,
                         bevel=0.004)
            f.rotation_euler = (0, math.pi / 2 - curl * (0.8 + 0.1 * i), 0)
        thumb = cylinder("Thumb", (0.10 * s, 0.05 * s, -0.004), 0.01 * s, 0.06 * s, skin, c,
                         axis="X", verts=10, pass_index=P, parent=root, bevel=0.004)
        thumb.rotation_euler = (0, math.pi / 2, math.radians(35))
        for obj in [root] + list(root.children):
            obj.hide_render = True
        self.hand_rng = rng
        return root

    def _apply_hand(self, t):
        if self.hand is None:
            return False
        reach, ev = 0.0, None
        for e, rng in zip(self.events, self.rngs):
            if e["type"] == "hand" and e["t0"] <= t < e["t1"]:
                reach = math.sin(math.pi * (t - e["t0"]) / max(e["t1"] - e["t0"], 1e-6))
                ev = e
        visible = reach > 0.02
        for obj in [self.hand] + list(self.hand.children):
            obj.hide_render = not visible
        if not visible:
            return False
        rng = random.Random(int((ev.get("params") or {}).get("seed", 0)))
        rig = self.ctx["rig"]
        hx, hy = rig.plate_half
        target = rig.bed.matrix_world @ Vector((rng.uniform(-0.6, 0.6) * hx,
                                                rng.uniform(-0.7, 0.3) * hy, 0.012))
        out = Vector((rng.uniform(-0.6, 0.6), -1.0, rng.uniform(0.35, 0.9))).normalized()
        reach = min(1.0, reach * rng.uniform(1.0, 1.6))
        tip = target + out * (0.45 * (1 - reach) + 0.02)
        rot = Vector((1, 0, 0)).rotation_difference(out).to_matrix().to_4x4()
        roll = Matrix.Rotation(rng.uniform(-0.4, 0.4), 4, "X")
        self.hand.matrix_world = Matrix.Translation(tip) @ rot @ roll
        return True

    # -- print time / park ------------------------------------------------------

    def print_time(self, t):
        for e in self.events:
            if e["type"] == "part_removed" and t >= e["t0"]:
                if t < e["t1"]:
                    return None
                return t - e["t1"]
        return t

    def park_position(self):
        hx, hy = self.ctx["rig"].plate_half
        return (0.9 * hx, 0.85 * hy, 0.03)

    def apply(self, t, nozzle):
        self._apply_lights(t)
        self._apply_doors(t)
        self._apply_hand(t)
        return [e["type"] for e in self.events if e["t0"] <= t < e["t1"]]
