"""Printer models.

Every builder returns a `Rig`: the static frame plus a few empties that move
(bed, gantry, carriage...). `rig.pose(x, y[, z])` places the moving parts so
the nozzle tip sits at bed-local (x, y, z), z defaulting to `rig.height`;
`rig.movers` are the empties to keyframe for motion blur. Anything whose shape
depends on Z (e.g. the Voron's Z belts) is updated inside the pose function.
Toolheads are modelled with the nozzle tip at the origin of `rig.carriage`, and
prints are parented to `rig.bed`, whose origin is the centre of the plate's top
surface.
"""

import math

import bpy

from .materials import emission, principled
from .util import (FPS, PASS_ENVIRONMENT, box, catmull, cylinder, empty, extrusion, fan,
                   point_light, rounded_slab, tube)

PI = math.pi


class Rig:
    def __init__(self, bed, carriage, plate_half, movers, pose, height):
        self.bed, self.carriage = bed, carriage
        self.plate_half = plate_half
        self.movers = movers
        self._pose = pose
        self.height = height

    def pose(self, x, y, z=None):
        self._pose(x, y, self.height if z is None else z)

    def animate(self, nozzle, direction, speed):
        """Key frames 0..2 so the nozzle is at `nozzle` on frame 1, moving
        along `direction` at `speed` m/s."""
        velocity = (direction.x * speed, direction.y * speed, 0.0)
        self.animate_motion((nozzle.x, nozzle.y, self.height), velocity)

    def animate_motion(self, position, velocity):
        """Key frames 0..2 so the nozzle is at bed-local `position` (m) on
        frame 1, moving at `velocity` (m/s). Replaces earlier keys, so it can
        be called once per rendered keyframe."""
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
        for obj in self.movers:
            obj.animation_data_clear()
        for frame in (0, 1, 2):
            k = (frame - 1) / FPS
            self.pose(*(position[i] + velocity[i] * k for i in range(3)))
            for obj in self.movers:
                obj.keyframe_insert("location", frame=frame)
        self.pose(*position)


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

def hotend(m, c, carriage, sock=True, block_y=0.002):
    box("HeaterBlock", (0, block_y, 0.0075), (0.016, 0.018, 0.009), m["aluminium"], c,
        bevel=0.001, parent=carriage)
    if sock:
        box("SiliconeSock", (0, block_y, 0.0075), (0.0175, 0.0195, 0.0095), m["sock"], c,
            bevel=0.002, parent=carriage)
    cylinder("NozzleHex", (0, 0, 0.0025), 0.0035, 0.002, m["brass"], c, verts=6,
             parent=carriage)
    cylinder("NozzleTip", (0, 0, 0.0008), 0.0006, 0.0016, m["brass"], c, verts=24,
             cone_r2=0.0022, parent=carriage)


def plate(m, c, bed, half, corner=0.006, tab=True):
    rounded_slab("BuildPlate", (0, 0, -0.0003), half, corner, 0.0006, m["plate"], c,
                 parent=bed)
    if tab:
        box("PlateTab", (-half[0] * 0.7, -half[1] - 0.005, -0.0003), (0.022, 0.012, 0.0006),
            m["plate_edge"], c, bevel=0.002, parent=bed)


# ---------------------------------------------------------------------------
# Toolheads (nozzle tip at carriage origin, front faces -Y)
# ---------------------------------------------------------------------------

def toolhead_bambu(m, c, carriage):
    """Boxy enclosed-printer toolhead with a removable front cover."""
    box("ToolheadBody", (0, 0.004, 0.052), (0.058, 0.05, 0.072), m["body"], c,
        bevel=0.006, parent=carriage)
    box("ToolheadCover", (0, -0.0225, 0.054), (0.054, 0.006, 0.064), m["cover"], c,
        bevel=0.005, parent=carriage)
    box("CoolingDuct", (0, -0.014, 0.012), (0.05, 0.024, 0.012), m["body"], c,
        bevel=0.004, parent=carriage)
    box("DuctLeft", (-0.022, 0.0, 0.010), (0.01, 0.03, 0.01), m["body"], c, bevel=0.003,
        parent=carriage)
    fan("HotendFan", (0, -0.0262, 0.058), 0.020, "-Y", m, c, parent=carriage)
    for i in range(4):
        box(f"SideVent{i}", (-0.0292, 0.004, 0.032 + i * 0.006), (0.0012, 0.03, 0.0025),
            m["void"], c, parent=carriage)
    hotend(m, c, carriage)
    ptfe = catmull([(0.008, 0.012, 0.088), (0.010, 0.02, 0.11),
                    (0.02, 0.07, 0.125), (0.04, 0.16, 0.115)])
    tube("PTFETube", [ptfe], 0.002, m["ptfe"], c, parent=carriage)
    cable = catmull([(-0.012, 0.02, 0.085), (-0.014, 0.035, 0.105),
                     (-0.02, 0.09, 0.118), (-0.04, 0.17, 0.10)])
    tube("CableBundle", [cable], 0.0035, m["rubber"], c, parent=carriage)


def toolhead_nextruder(m, c, carriage):
    """Prusa MK4-style: orange shell, planetary gearbox up front, side fan."""
    box("ToolheadBody", (0, 0.012, 0.058), (0.05, 0.046, 0.08), m["body"], c,
        bevel=0.004, parent=carriage)
    box("ToolheadShell", (0, -0.016, 0.06), (0.058, 0.022, 0.07), m["accent"], c,
        bevel=0.007, parent=carriage)
    cylinder("Gearbox", (0, -0.028, 0.083), 0.019, 0.012, m["accent"], c, axis="Y",
             verts=40, bevel=0.002, parent=carriage)
    cylinder("GearboxCap", (0, -0.0345, 0.083), 0.009, 0.002, m["body"], c, axis="Y",
             parent=carriage)
    box("FanShroud", (0, -0.017, 0.016), (0.05, 0.024, 0.014), m["accent"], c,
        bevel=0.004, parent=carriage)
    fan("HotendFan", (-0.0292, 0.004, 0.055), 0.016, "-X", m, c, parent=carriage)
    box("Logo", (0.012, -0.0272, 0.04), (0.014, 0.0006, 0.006), m["body"], c,
        parent=carriage)
    hotend(m, c, carriage)
    cable = catmull([(0.0, 0.03, 0.095), (0.0, 0.05, 0.13), (0.03, 0.09, 0.2),
                     (0.08, 0.1, 0.3)])
    tube("TextileSleeve", [cable], 0.005, m["cable"], c, parent=carriage)


def toolhead_stealthburner(m, c, carriage):
    """Voron Stealthburner: red cowl, black core, glowing logo and nozzle LEDs."""
    box("ToolheadBody", (0, 0.012, 0.062), (0.052, 0.046, 0.085), m["body"], c,
        bevel=0.005, parent=carriage)
    box("Cowl", (0, -0.02, 0.06), (0.058, 0.026, 0.09), m["accent"], c, bevel=0.009,
        parent=carriage)
    box("CowlInsert", (0, -0.0335, 0.07), (0.03, 0.002, 0.05), m["body"], c,
        bevel=0.004, parent=carriage)
    logo = emission("StealthburnerLED", (0.85, 0.9, 1.0), 25.0)
    box("LogoLED", (0, -0.0347, 0.078), (0.012, 0.0008, 0.016), logo, c, bevel=0.002,
        parent=carriage)
    for side in (-1, 1):
        box(f"Duct{side:+d}", (side * 0.02, -0.008, 0.016), (0.012, 0.03, 0.024),
            m["accent"], c, bevel=0.003, parent=carriage,
            rotation=(0, side * math.radians(12), 0))
        box(f"NozzleLED{side:+d}", (side * 0.008, -0.026, 0.025), (0.004, 0.003, 0.0008),
            logo, c, parent=carriage)
        point_light(f"NozzleLight{side:+d}", c, (side * 0.008, -0.024, 0.022), 0.25,
                    (1.0, 0.97, 0.92), radius=0.002, parent=carriage)
    fan("HotendFan", (-0.0262, 0.008, 0.062), 0.017, "-X", m, c, parent=carriage)
    box("Extruder", (0, 0.016, 0.118), (0.042, 0.04, 0.03), m["body"], c, bevel=0.004,
        parent=carriage)
    cylinder("ExtruderKnob", (0.023, 0.016, 0.118), 0.008, 0.006, m["accent"], c,
             axis="X", parent=carriage)
    hotend(m, c, carriage)
    umbilical = catmull([(0, 0.03, 0.13), (0, 0.06, 0.17), (0.02, 0.14, 0.24),
                         (0.05, 0.22, 0.27)])
    tube("Umbilical", [umbilical], 0.006, m["cable"], c, parent=carriage)


def toolhead_ender(m, c, carriage):
    """Stock Ender-3: sheet-metal shroud with a front fan, side blower."""
    box("CarriagePlate", (0, 0.022, 0.045), (0.07, 0.004, 0.07), m["anodized"], c,
        bevel=0.003, parent=carriage)
    box("Shroud", (0, -0.002, 0.043), (0.045, 0.042, 0.05), m["body"], c, bevel=0.0015,
        parent=carriage)
    fan("HotendFan", (0, -0.0232, 0.045), 0.017, "-Y", m, c, parent=carriage)
    box("Blower", (0.031, -0.002, 0.04), (0.016, 0.05, 0.05), m["accent"], c,
        bevel=0.004, parent=carriage)
    box("BlowerDuct", (0.022, -0.004, 0.011), (0.014, 0.02, 0.014), m["accent"], c,
        bevel=0.003, parent=carriage, rotation=(0, math.radians(-30), 0))
    hotend(m, c, carriage, sock=False)
    for i, col in enumerate(((0.6, 0.02, 0.02), (0.9, 0.9, 0.9))):
        wire = catmull([(0.006 + i * 0.002, 0.008, 0.008), (0.012, 0.015, 0.018),
                        (0.014, 0.03, 0.05)], per_seg=6)
        tube(f"HeaterWire{i}", [wire], 0.0007, principled(f"Wire{i}", col, 0.5), c,
             parent=carriage)
    bowden = catmull([(0, 0.0, 0.068), (0, 0.0, 0.1), (-0.04, 0.02, 0.2),
                      (-0.15, 0.03, 0.26)])
    tube("BowdenTube", [bowden], 0.002, m["ptfe"], c, parent=carriage)


# ---------------------------------------------------------------------------
# Enclosed CoreXY (Bambu P1S / X1C layout): bed moves in Z
# ---------------------------------------------------------------------------

BAMBU = {"nozzle_z": 0.30, "gantry_z": 0.37, "y_front": -0.185, "y_back": 0.195,
         "plate_center": (0.0, 0.012), "plate": 0.257}


def build_bambu(m, colls, height):
    c = colls["Printer"]
    nz, gz = BAMBU["nozzle_z"], BAMBU["gantry_z"]
    yf, yb = BAMBU["y_front"], BAMBU["y_back"]
    px, py = BAMBU["plate_center"]
    half = BAMBU["plate"] / 2

    bed = empty("Bed", c)
    plate(m, c, bed, (half, half))
    box("Heatbed", (0, 0, -0.0031), (0.262, 0.262, 0.005), m["anodized"], c,
        bevel=0.0015, parent=bed)
    box("BedCarrier", (0, 0.01, -0.012), (0.22, 0.24, 0.012), m["sheet"], c,
        bevel=0.002, parent=bed)

    screws = [(-0.168, 0.168), (0.168, 0.168), (0.168, -0.152)]
    for i, (sx, sy) in enumerate(screws):
        cylinder(f"LeadScrew{i}", (sx, sy, 0.165), 0.004, 0.33, m["screw"], c, verts=20)
        gy = sy - 0.022 if sy > 0 else sy + 0.022
        cylinder(f"ZRod{i}", (sx, gy, 0.165), 0.004, 0.33, m["steel"], c, verts=20)
        inner = px + math.copysign(0.105, sx)
        box(f"BedArm{i}", ((sx + inner) / 2 - px, (sy + gy) / 2 - py, -0.012),
            (abs(sx - inner) + 0.02, 0.04, 0.01), m["sheet"], c, bevel=0.002, parent=bed)
        box(f"ZNut{i}", (sx - px, (sy + gy) / 2 - py, -0.022), (0.022, 0.05, 0.016),
            m["body"], c, bevel=0.002, parent=bed)
        box(f"ZTop{i}", (sx, (sy + gy) / 2, 0.335), (0.022, 0.05, 0.01), m["sheet"], c,
            bevel=0.002)

    for side in (-1, 1):
        xr = side * 0.172
        cylinder(f"YRod{side:+d}", (xr, (yf + yb) / 2, gz), 0.004, yb - yf - 0.02,
                 m["carbon"], c, axis="Y", verts=24)
        box(f"FrontIdler{side:+d}", (xr, yf + 0.016, gz), (0.03, 0.022, 0.034), m["body"],
            c, bevel=0.003)
        box(f"Motor{side:+d}", (xr - side * 0.004, yb - 0.026, gz + 0.012),
            (0.042, 0.042, 0.042), m["motor"], c, bevel=0.003)
        cylinder(f"MotorPulley{side:+d}", (xr - side * 0.004, yb - 0.026, gz - 0.016),
                 0.006, 0.014, m["steel"], c, verts=20)
        for k, dx in enumerate((-0.011, 0.011)):
            box(f"YBelt{side:+d}{k}", (xr + dx, (yf + yb) / 2 - 0.004, gz - 0.012),
                (0.0012, yb - yf - 0.05, 0.006), m["rubber"], c)

    gantry = empty("GantryY", c)
    carriage = empty("Carriage", c, parent=gantry)
    for side in (-1, 1):
        box(f"YCarriage{side:+d}", (side * 0.172, 0, gz), (0.03, 0.05, 0.03), m["body"], c,
            bevel=0.004, parent=gantry)
    cylinder("XRodUpper", (0, 0.010, gz + 0.008), 0.004, 0.33, m["carbon"], c, axis="X",
             verts=24, parent=gantry)
    cylinder("XRodLower", (0, 0.010, gz - 0.016), 0.004, 0.33, m["carbon"], c, axis="X",
             verts=24, parent=gantry)
    for k, dz in enumerate((0.0, -0.008)):
        box(f"XBelt{k}", (0, 0.022, gz + dz), (0.32, 0.0012, 0.006), m["rubber"], c,
            parent=gantry)
    toolhead_bambu(m, c, carriage)

    def pose(x, y, h):
        bed.location = (px, py, nz - h)
        gantry.location = (0, py + y, 0)
        carriage.location = (px + x, 0, nz)

    return Rig(bed, carriage, (half, half), [bed, gantry, carriage], pose, height)


# ---------------------------------------------------------------------------
# Prusa MK4: bed slinger, bed moves in Y, X axis moves in Z
# ---------------------------------------------------------------------------

def build_prusa_mk4(m, colls, height):
    c = colls["Printer"]
    bed_z = 0.072
    half = (0.125, 0.105)

    for s in (-1, 1):
        extrusion(f"YFrame{s:+d}", (s * 0.09, 0, 0.022), 0.4, "Y", m["frame"], m["void"],
                  c, profile=(0.02, 0.04))
        cylinder(f"YRod{s:+d}", (s * 0.072, 0, 0.05), 0.004, 0.38, m["steel"], c,
                 axis="Y", verts=20)
        for e in (-1, 1):
            cylinder(f"Foot{s:+d}{e:+d}", (s * 0.15, e * 0.2, 0.004), 0.012, 0.008,
                     m["rubber"], c)
    for e, name in ((-1, "Front"), (1, "Rear")):
        box(f"{name}Plate", (0, e * 0.206, 0.04), (0.32, 0.006, 0.08), m["frame"], c,
            bevel=0.0015)
    box("YBelt", (0, 0, 0.056), (0.006, 0.38, 0.0012), m["rubber"], c)
    box("YMotor", (0, -0.226, 0.04), (0.042, 0.034, 0.042), m["motor"], c, bevel=0.003)

    bed = empty("Bed", c)
    plate(m, c, bed, half, corner=0.005)
    box("Heatbed", (0, 0, -0.0021), (0.255, 0.215, 0.003), m["pcb"], c, bevel=0.0008,
        parent=bed)
    box("BedCarrier", (0, 0, -0.012), (0.19, 0.17, 0.004), m["frame"], c, bevel=0.001,
        parent=bed)
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(f"BedBearing{sx:+d}{sy:+d}", (sx * 0.072, sy * 0.06, -0.02),
                (0.022, 0.03, 0.014), m["accent"], c, bevel=0.003, parent=bed)

    # The iconic one-piece aluminium Z frame, with its big window.
    fy = 0.078
    box("FrameLeft", (-0.2, fy, 0.24), (0.05, 0.006, 0.44), m["frame"], c, bevel=0.0015)
    box("FrameRight", (0.2, fy, 0.24), (0.05, 0.006, 0.44), m["frame"], c, bevel=0.0015)
    box("FrameTop", (0, fy, 0.435), (0.45, 0.006, 0.05), m["frame"], c, bevel=0.0015)
    box("FrameBottom", (0, fy, 0.04), (0.45, 0.006, 0.08), m["frame"], c, bevel=0.0015)
    for s in (-1, 1):
        box(f"ZMotor{s:+d}", (s * 0.19, 0.046, 0.03), (0.042, 0.042, 0.042), m["motor"],
            c, bevel=0.003)
        cylinder(f"ZScrew{s:+d}", (s * 0.19, 0.046, 0.24), 0.004, 0.37, m["screw"], c,
                 verts=20)
        cylinder(f"ZRod{s:+d}", (s * 0.165, 0.05, 0.24), 0.004, 0.37, m["steel"], c,
                 verts=20)
        box(f"ZTop{s:+d}", (s * 0.18, 0.055, 0.428), (0.055, 0.04, 0.024), m["accent"], c,
            bevel=0.003)

    xaxis = empty("XAxis", c)
    carriage = empty("Carriage", c, parent=xaxis)
    for k, dz in enumerate((0.032, 0.077)):
        cylinder(f"XRod{k}", (0, 0.034, dz), 0.004, 0.37, m["steel"], c, axis="X",
                 verts=20, parent=xaxis)
    box("XBelt", (0, 0.03, 0.055), (0.34, 0.0012, 0.006), m["rubber"], c, parent=xaxis)
    for s in (-1, 1):
        box(f"XEnd{s:+d}", (s * 0.178, 0.045, 0.055), (0.048, 0.05, 0.078), m["accent"], c,
            bevel=0.004, parent=xaxis)
    box("XMotor", (-0.224, 0.04, 0.055), (0.042, 0.042, 0.042), m["motor"], c,
        bevel=0.003, parent=xaxis)
    toolhead_nextruder(m, c, carriage)

    # LCD knob unit at the front left.
    lcd = box("LCDHousing", (-0.11, -0.232, 0.055), (0.13, 0.03, 0.075), m["accent"], c,
              bevel=0.005, rotation=(math.radians(-18), 0, 0))
    screen = emission("LCDScreen", (0.95, 0.55, 0.2), 1.2)
    box("LCDScreen", (-0.02, -0.0152, 0.008), (0.07, 0.0006, 0.042), screen, c,
        parent=lcd)
    cylinder("LCDKnob", (0.042, -0.019, -0.012), 0.009, 0.01, m["body"], c, axis="Y",
             parent=lcd)

    def pose(x, y, h):
        bed.location = (0, -y, bed_z)
        xaxis.location = (0, 0, bed_z + h)
        carriage.location = (x, 0, 0)

    return Rig(bed, carriage, half, [bed, xaxis, carriage], pose, height)


# ---------------------------------------------------------------------------
# Creality Ender-3: V-slot frame, glass bed on springs, single Z screw
# ---------------------------------------------------------------------------

def build_ender3(m, colls, height):
    c = colls["Printer"]
    bed_z = 0.098
    half = (0.1175, 0.1175)

    for s in (-1, 1):
        extrusion(f"BaseRail{s:+d}", (s * 0.16, 0, 0.02), 0.44, "Y", m["frame"], m["void"],
                  c, profile=(0.02, 0.04))
        extrusion(f"BaseBar{s:+d}", (0, s * 0.21, 0.02), 0.30, "X", m["frame"], m["void"],
                  c, profile=(0.02, 0.04))
    extrusion("YRail", (0, 0, 0.02), 0.42, "Y", m["frame"], m["void"], c,
              profile=(0.04, 0.04))

    bed = empty("Bed", c)
    rounded_slab("BuildPlate", (0, 0, -0.002), half, 0.002, 0.004, m["plate"], c,
                 parent=bed)
    box("Heatbed", (0, 0, -0.0055), (0.235, 0.235, 0.003), m["pcb"], c, parent=bed)
    box("BedCarriage", (0, 0, -0.03), (0.22, 0.2, 0.004), m["aluminium"], c,
        bevel=0.001, parent=bed)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cylinder(f"Spring{sx:+d}{sy:+d}", (sx * 0.1, sy * 0.1, -0.018), 0.0048, 0.02,
                     m["spring"], c, verts=16, parent=bed)
            cylinder(f"Knob{sx:+d}{sy:+d}", (sx * 0.1, sy * 0.1, -0.039), 0.014, 0.01,
                     m["motor"], c, verts=24, parent=bed)
    # Binder clips holding the glass.
    for sx in (-1, 1):
        for y in (-0.06, 0.06):
            box(f"Clip{sx:+d}{y:+.2f}", (sx * 0.119, y, -0.0035), (0.01, 0.028, 0.013),
                m["rubber"], c, bevel=0.001, parent=bed)
            handle = [(sx * 0.117, y - 0.009, 0.003), (sx * 0.128, y - 0.009, 0.022),
                      (sx * 0.128, y + 0.009, 0.022), (sx * 0.117, y + 0.009, 0.003)]
            tube(f"ClipHandle{sx:+d}{y:+.2f}", [handle], 0.0007, m["steel"], c,
                 parent=bed)

    for s in (-1, 1):
        extrusion(f"Upright{s:+d}", (s * 0.22, 0.03, 0.255), 0.43, "Z", m["frame"],
                  m["void"], c, profile=(0.02, 0.04))
    extrusion("TopBar", (0, 0.03, 0.48), 0.46, "X", m["frame"], m["void"], c,
              profile=(0.02, 0.02))
    box("ZMotor", (-0.19, 0.03, 0.06), (0.042, 0.042, 0.042), m["motor"], c, bevel=0.003)
    cylinder("ZScrew", (-0.19, 0.03, 0.27), 0.004, 0.38, m["screw"], c, verts=20)
    box("PSU", (0.265, 0.06, 0.12), (0.05, 0.2, 0.11), m["anodized"], c, bevel=0.002)
    box("Extruder", (-0.245, 0.03, 0.33), (0.03, 0.045, 0.045), m["motor"], c,
        bevel=0.003)
    box("ExtruderArm", (-0.245, 0.0, 0.345), (0.03, 0.012, 0.03), m["accent"], c,
        bevel=0.002)
    lcd = box("LCD", (0.15, -0.24, 0.07), (0.1, 0.02, 0.065), m["body"], c, bevel=0.003,
              rotation=(math.radians(-15), 0, 0))
    box("LCDScreen", (0, -0.0102, 0.005), (0.07, 0.0006, 0.04),
        emission("LCDScreen", (0.25, 0.45, 1.0), 1.5), c, parent=lcd)

    xaxis = empty("XAxis", c)
    carriage = empty("Carriage", c, parent=xaxis)
    extrusion("XBeam", (0, 0.034, 0.045), 0.43, "X", m["frame"], m["void"], c,
              profile=(0.02, 0.02), parent=xaxis)
    box("XBelt", (0, 0.0235, 0.045), (0.38, 0.0012, 0.006), m["rubber"], c, parent=xaxis)
    box("XEndLeft", (-0.215, 0.026, 0.045), (0.06, 0.005, 0.08), m["anodized"], c,
        bevel=0.002, parent=xaxis)
    box("XMotor", (-0.205, 0.0, 0.045), (0.042, 0.042, 0.042), m["motor"], c, bevel=0.003,
        parent=xaxis)
    box("XEndRight", (0.215, 0.026, 0.045), (0.05, 0.005, 0.07), m["anodized"], c,
        bevel=0.002, parent=xaxis)
    toolhead_ender(m, c, carriage)

    def pose(x, y, h):
        bed.location = (0, -y, bed_z)
        xaxis.location = (0, 0, bed_z + h)
        carriage.location = (x, 0, 0)

    return Rig(bed, carriage, half, [bed, xaxis, carriage], pose, height)


# ---------------------------------------------------------------------------
# Voron 2.4: fixed bed, flying gantry moves in Z
# ---------------------------------------------------------------------------

VORON = {"half_w": 0.24, "top": 0.56, "deck": 0.12, "bed_z": 0.16}


def build_voron(m, colls, height):
    c, env = colls["Printer"], colls["Environment"]
    w, top, deck, bed_z = VORON["half_w"], VORON["top"], VORON["deck"], VORON["bed_z"]
    half = (0.15, 0.15)

    for sx in (-1, 1):
        for sy in (-1, 1):
            extrusion(f"Post{sx:+d}{sy:+d}", (sx * w, sy * w, top / 2), top, "Z",
                      m["frame"], m["void"], c)
    for z in (0.01, deck - 0.01, top - 0.01):
        for s in (-1, 1):
            extrusion(f"RailX{s:+d}_{z:.2f}", (0, s * w, z), 2 * w - 0.02, "X",
                      m["frame"], m["void"], c)
            extrusion(f"RailY{s:+d}_{z:.2f}", (s * w, 0, z), 2 * w - 0.02, "Y",
                      m["frame"], m["void"], c)
    box("Deck", (0, 0, deck + 0.002), (2 * w - 0.02, 2 * w - 0.02, 0.003), m["sheet"], c)

    bed = empty("Bed", c, (0, 0, bed_z))
    plate(m, c, bed, half, corner=0.004)
    box("BedAlu", (0, 0, -0.0036), (0.305, 0.305, 0.006), m["aluminium"], c,
        bevel=0.0005, parent=bed)
    box("BedHeater", (0, 0, -0.0072), (0.27, 0.27, 0.0014), m["silicone_heater"], c,
        parent=bed)
    for s in (-1, 1):
        extrusion(f"BedRail{s:+d}", (s * 0.1, 0, -0.025), 0.43, "Y", m["frame"], m["void"],
                  c, parent=bed)

    belts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            box(f"ZMotor{sx:+d}{sy:+d}", (sx * 0.195, sy * 0.195, deck + 0.03),
                (0.042, 0.042, 0.048), m["motor"], c, bevel=0.003)
            box(f"ZMount{sx:+d}{sy:+d}", (sx * 0.195, sy * 0.195, deck + 0.06),
                (0.05, 0.05, 0.012), m["accent"], c, bevel=0.003)
            belts.append(box(f"ZBelt{sx:+d}{sy:+d}", (sx * 0.205, sy * 0.195, 0.0),
                             (0.0012, 0.006, 1.0), m["rubber"], c))

    zstage = empty("ZStage", c)
    ybeam = empty("YBeam", c, parent=zstage)
    carriage = empty("Carriage", c, parent=ybeam)
    for s in (-1, 1):
        extrusion(f"GantryY{s:+d}", (s * 0.205, 0, 0.055), 0.42, "Y", m["frame"], m["void"],
                  c, parent=zstage)
        box(f"GantryRail{s:+d}", (s * 0.205, 0, 0.0665), (0.012, 0.4, 0.004), m["steel"], c,
            parent=zstage)
        for e in (-1, 1):
            box(f"GantryJoint{s:+d}{e:+d}", (s * 0.205, e * 0.205, 0.05), (0.05, 0.05, 0.045),
                m["accent"], c, bevel=0.004, parent=zstage)
        box(f"ABMotor{s:+d}", (s * 0.205, 0.215, 0.1), (0.042, 0.042, 0.048), m["motor"], c,
            bevel=0.003, parent=zstage)
        box(f"YBelt{s:+d}", (s * 0.19, 0, 0.075), (0.0012, 0.38, 0.006), m["rubber"], c,
            parent=zstage)
    extrusion("XBeam", (0, 0.046, 0.055), 0.39, "X", m["frame"], m["void"], c,
              parent=ybeam)
    box("XRail", (0, 0.0355, 0.055), (0.37, 0.002, 0.012), m["steel"], c, parent=ybeam)
    for s in (-1, 1):
        box(f"XYJoint{s:+d}", (s * 0.205, 0.046, 0.068), (0.05, 0.06, 0.035), m["accent"], c,
            bevel=0.004, parent=ybeam)
    box("XBelt", (0, 0.046, 0.078), (0.38, 0.006, 0.0012), m["rubber"], c, parent=ybeam)
    toolhead_stealthburner(m, c, carriage)

    # Tinted side/back panels, front doors with printed handles, solid top.
    thick = 0.003
    for sx in (-1, 1):
        box(f"PanelSide{sx:+d}", (sx * (w + 0.0115), 0, top / 2), (thick, 2 * w, top - 0.02),
            m["acrylic_smoke"], env, PASS_ENVIRONMENT)
    box("PanelBack", (0, w + 0.0115, top / 2), (2 * w, thick, top - 0.02),
        m["acrylic_smoke"], env, PASS_ENVIRONMENT)
    for sx in (-1, 1):
        box(f"Door{sx:+d}", (sx * w / 2, -w - 0.0125, top / 2), (w - 0.004, thick, top - 0.02),
            m["acrylic_smoke"], env, PASS_ENVIRONMENT)
        box(f"DoorHandle{sx:+d}", (sx * 0.03, -w - 0.022, top / 2), (0.012, 0.016, 0.07),
            m["accent"], env, PASS_ENVIRONMENT, bevel=0.003)
    box("TopPanel", (0, 0, top + 0.0115), (2 * w, 2 * w, thick), m["acrylic_smoke"], env,
        PASS_ENVIRONMENT)

    def pose(x, y, h):
        zstage.location = (0, 0, bed_z + h)
        ybeam.location = (0, y, 0)
        carriage.location = (x, 0, 0)
        # Z belts run from the deck drives up to the gantry joints.
        zb = bed_z + h + 0.04 - (deck + 0.066)
        for belt in belts:
            belt.location.z = deck + 0.066 + zb / 2
            belt.scale.z = zb

    pose(0.0, 0.0, height)

    return Rig(bed, carriage, half, [zstage, ybeam, carriage], pose, height)


# ---------------------------------------------------------------------------
# Compact cantilever bed slinger (A1 mini layout): one Z column, X arm
# ---------------------------------------------------------------------------

def build_cantilever(m, colls, height):
    c = colls["Printer"]
    bed_z = 0.07
    half = (0.09, 0.09)

    box("Base", (0.02, 0.0, 0.025), (0.33, 0.36, 0.05), m["cover"], c, bevel=0.008)
    box("BaseSlot", (0.0, 0.0, 0.0505), (0.03, 0.3, 0.002), m["void"], c)
    box("Column", (-0.165, 0.07, 0.23), (0.045, 0.06, 0.36), m["cover"], c, bevel=0.006)
    box("ColumnSlot", (-0.142, 0.07, 0.23), (0.002, 0.012, 0.34), m["void"], c)
    box("ColumnCap", (-0.165, 0.07, 0.415), (0.05, 0.066, 0.012), m["body"], c,
        bevel=0.004)
    box("Screen", (0.12, -0.181, 0.03), (0.06, 0.002, 0.03),
        emission("StatusScreen", (0.9, 0.95, 1.0), 0.6), c)
    status = emission("StatusLED", (0.2, 1.0, 0.3), 30.0)
    box("StatusLED", (0.165, -0.181, 0.03), (0.004, 0.002, 0.004), status, c)

    bed = empty("Bed", c)
    plate(m, c, bed, half, corner=0.004)
    box("Heatbed", (0, 0, -0.003), (0.185, 0.185, 0.005), m["anodized"], c, bevel=0.001,
        parent=bed)
    box("BedCarrier", (0, 0, -0.012), (0.12, 0.14, 0.012), m["body"], c, bevel=0.002,
        parent=bed)

    xaxis = empty("XAxis", c)
    carriage = empty("Carriage", c, parent=xaxis)
    box("Arm", (-0.01, 0.046, 0.055), (0.28, 0.032, 0.042), m["cover"], c, bevel=0.006,
        parent=xaxis)
    box("ArmSlot", (-0.01, 0.0295, 0.055), (0.25, 0.002, 0.008), m["void"], c,
        parent=xaxis)
    box("ArmEnd", (0.135, 0.046, 0.055), (0.012, 0.036, 0.046), m["body"], c, bevel=0.004,
        parent=xaxis)
    toolhead_bambu(m, c, carriage)

    def pose(x, y, h):
        bed.location = (0, -y, bed_z)
        xaxis.location = (0, 0, bed_z + h)
        carriage.location = (x, 0, 0)

    return Rig(bed, carriage, half, [bed, xaxis, carriage], pose, height)

