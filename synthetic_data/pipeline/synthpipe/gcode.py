"""Streaming G-code -> CONTRACTS.md §3 timeline.

Startup heating/dwells/home are not timed. Explicit purge/prime/wipe extrusion
before the first proper extrusion is retained at negative times, with intervening
moves; other startup moves are dropped. Print proper means positive-E motion in
a layer or a recognized non-purge feature (unannotated files start at their first
extrusion). G92 changes logical coordinates, never the physical nozzle position.
Negative E is emitted on stationary rows even when combined with XYZ motion.
Arcs in G17/G18/G19 planes use <=1 mm chords and <=0.02 mm sagitta.

For slicer comparisons, the raw header estimate and ratio are retained; a
separate print-only estimate uses the latest M73 remaining minutes at the first
print extrusion when no explicit model-time header exists. This avoids counting
Prusa startup heating time as model time (M73 has minute-level precision).

Time uses acceleration-limited trapezoids and junction-deviation 0.02 mm (or
emitted M205 J / Klipper square-corner velocity), with conservative local
reachability and emitted minimum-cruise/pseudo-acceleration limits. No calibration to slicer time is applied.
Timeline queries intentionally interpolate linearly, per the interchange contract.
"""

import json
import logging
import math
import re
import time
from array import array
from pathlib import Path

import numpy as np

LOG = logging.getLogger(__name__)
KINDS = {
    0: "travel",
    1: "outer wall",
    2: "inner wall",
    3: "sparse infill",
    4: "solid infill",
    5: "top/bottom surface",
    6: "support",
    7: "support interface",
    8: "skirt/brim",
    9: "purge/prime/wipe",
    10: "other extrusion",
}
_FEATURES = {
    "outer wall": 1,
    "external perimeter": 1,
    "overhang wall": 1,
    "overhang perimeter": 1,
    "inner wall": 2,
    "perimeter": 2,
    "internal perimeter": 2,
    "sparse infill": 3,
    "internal infill": 3,
    "infill": 3,
    "internal solid infill": 4,
    "solid infill": 4,
    "bridge": 4,
    "bridge infill": 4,
    "internal bridge": 4,
    "internal bridge infill": 4,
    "top surface": 5,
    "bottom surface": 5,
    "top solid infill": 5,
    "bottom solid infill": 5,
    "support": 6,
    "support material": 6,
    "support transition": 6,
    "support interface": 7,
    "support material interface": 7,
    "skirt": 8,
    "brim": 8,
    "skirt/brim": 8,
    "prime tower": 9,
    "wipe tower": 9,
    "purge": 9,
    "prime": 9,
    "purge line": 9,
    "prime line": 9,
    "wipe": 9,
    "custom": 9,
    "custom gcode": 9,
    "gap infill": 10,
    "gap fill": 10,
    "ironing": 5,
}
_WORDS = re.compile(r"([A-Za-z])\s*([-+]?(?:\d*\.\d+|\d+\.?\d*))")
_NUMBERS = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)")


def _feature(name, unknown):
    name = name.strip().lower()
    if name in _FEATURES:
        return _FEATURES[name]
    if name not in unknown:
        LOG.warning("Unknown extrusion feature %r; using kind 10", name)
        unknown.add(name)
    return 10


def _words(text):
    parts = text.split()
    try:
        if len(parts) == 1 and ("E" in parts[0][1:] or "e" in parts[0][1:]):
            raise ValueError("Compact extrusion word")
        return {p[0]: float(p[1:]) for p in parts}
    except ValueError:
        return {key.upper(): float(val) for key, val in _WORDS.findall(text)}


def _estimate(text):
    text = text.split(";", 1)[0]
    units = re.findall(r"(\d+(?:\.\d+)?)\s*([dhms])", text.lower())
    if units:
        return sum(float(n) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[u] for n, u in units)
    try:
        return float(text.strip())
    except ValueError:
        return None


def _arc(start, end, words, clockwise, plane):
    axes, offsets = {
        "G17": ((0, 1), ("I", "J")),
        "G18": ((2, 0), ("K", "I")),
        "G19": ((1, 2), ("J", "K")),
    }[plane]
    a, b = axes
    x, y = start[a], start[b]
    ex, ey = end[a], end[b]
    if any(k in words for k in offsets):
        cx, cy = x + words.get(offsets[0], 0), y + words.get(offsets[1], 0)
    elif "R" in words:
        radius = abs(words["R"])
        dx, dy = ex - x, ey - y
        chord = math.hypot(dx, dy)
        if not chord or chord > 2 * radius + 1e-6:
            raise ValueError("Invalid radius arc")
        h = math.sqrt(max(0, radius * radius - chord * chord / 4))
        sign = (-1 if clockwise else 1) * (-1 if words["R"] < 0 else 1)
        cx, cy = (
            (x + ex) / 2 - sign * dy * h / chord,
            (y + ey) / 2 + sign * dx * h / chord,
        )
    else:
        raise ValueError("G2/G3 requires IJK or R")
    radius = math.hypot(x - cx, y - cy)
    if radius < 1e-9:
        raise ValueError("Zero radius arc")
    begin = math.atan2(y - cy, x - cx)
    finish = math.atan2(ey - cy, ex - cx)
    sweep = (finish - begin) % (2 * math.pi)
    if clockwise:
        sweep = -((begin - finish) % (2 * math.pi))
    if abs(sweep) < 1e-10:
        sweep = -2 * math.pi if clockwise else 2 * math.pi
    step = min(1 / radius, 2 * math.acos(max(-1, 1 - 0.02 / radius)))
    other = 3 - a - b
    n = max(
        1,
        math.ceil(abs(sweep) / step),
        math.ceil(math.hypot(radius * sweep, end[other] - start[other])),
    )
    points = []
    for j in range(1, n + 1):
        f = j / n
        p = [start[k] + (end[k] - start[k]) * f for k in range(3)]
        p[a], p[b] = (
            cx + radius * math.cos(begin + sweep * f),
            cy + radius * math.sin(begin + sweep * f),
        )
        points.append(p)
    points[-1] = end[:]
    return points


def _durations(p0, p1, e, feed, accel, junction_deviation=0.02, cruise_ratio=1.0):
    delta = p1.astype(np.float64) - p0
    length = np.linalg.norm(delta, axis=1)
    distance = np.where(length > 1e-9, length, np.abs(e))
    unit = delta / np.maximum(length[:, None], 1e-12)
    n = len(length)
    pseudo_accel = accel * cruise_ratio
    junction = np.zeros(n + 1)
    if n > 1:
        cosine = np.clip(np.einsum("ij,ij->i", unit[:-1], unit[1:]), -1, 1)
        s = np.sqrt(np.maximum(0, (1 + cosine) / 2))
        aj = accel * junction_deviation
        limit = np.minimum(aj[:-1], aj[1:]) * s / np.maximum(1 - s, 1e-12)
        bound = np.minimum(pseudo_accel[:-1] * distance[:-1], pseudo_accel[1:] * distance[1:])
        junction[1:-1] = np.sqrt(np.minimum(limit, bound))
        junction[1:-1] = np.minimum(junction[1:-1], np.minimum(feed[:-1], feed[1:]))
        junction[1:-1][(length[:-1] < 1e-9) | (length[1:] < 1e-9)] = 0
    vi, vo = junction[:-1], junction[1:]
    peak = np.minimum(
        feed, np.sqrt(np.maximum(0, pseudo_accel * distance + (vi * vi + vo * vo) / 2))
    )
    ramp_distance = (2 * peak * peak - vi * vi - vo * vo) / (2 * accel)
    return np.maximum(
        (2 * peak - vi - vo) / accel
        + np.maximum(0, distance - ramp_distance) / np.maximum(peak, 1e-9),
        1e-8,
    )


def parse_gcode(path, *, printer_profile=None):
    """Parse a file incrementally; return a Timeline (also exposes typed arrays).

    Unknown bed dimensions remain null with a warning, never guessed from a
    printer name. Header bed polygons define centre as their bounding-box centre.
    An adjacent slice.json may supply printer_profile, but is never required.
    """
    path = Path(path)
    # Prefer the first real layer over object declarations, which some slicers
    # emit in the header before their start macro. This short preflight normally
    # stops in the header/start block; unannotated files retain the old fallback.
    print_boundary = None
    first_label = None
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for number, raw in enumerate(stream, 1):
            command, _, comment = raw.partition(";")
            low = comment.strip().lower()
            if low in ("change_layer", "layer_change") or low.startswith("layer:"):
                print_boundary = number
                break
            if first_label is None and (
                low.startswith("start printing object, unique label id:")
                or command.strip().startswith("EXCLUDE_OBJECT_START ")
                or re.match(r"M486\s+S\s*\d+", command.strip())
            ):
                first_label = number
    if print_boundary is None:
        print_boundary = first_label
    purge_block = False
    saved_kind = None
    last_valid_height = 0.2
    cruise_settings = array("d")
    cruise_setting = 1.0
    junctions = array("d")
    junction_setting = 0.02
    object_ids = array("h")
    objects = []
    object_lookup = {}
    current_object = -1

    def object_index(key, name=None):
        if key not in object_lookup:
            if len(objects) >= 32768:
                raise ValueError("Too many labelled objects for int16 timeline")
            object_lookup[key] = len(objects)
            objects.append(dict(id=len(objects), name=name or str(key[1]),
                                center_mm=None, polygon_mm=None, bbox_mm=None, n_moves=0))
        index = object_lookup[key]
        if name is not None:
            objects[index]["name"] = name
        return index

    rows = array("d")  # flat packed numeric records, not Python objects per move
    pos = [0.0, 0.0, 0.0]
    offset = [0.0, 0.0, 0.0]
    epos = 0.0
    feed = 50.0
    absolute = True
    eabsolute = True
    plane = "G17"
    kind = 9
    height = 0.2
    width = 0.0
    layer = -1
    pending_layer = False
    layer_z = []
    current_z = None
    layer_explicit = False
    pending_start = None
    declared_z = None
    height_explicit = False
    accel = float("nan")
    travel_accel = float("nan")
    machine_accel = 5000.0
    extrusion_accel = float("nan")
    retract_accel = float("nan")
    axis_accel = [float("inf")] * 3
    axis_speed = [float("inf")] * 4
    filament = 1.75
    nozzle = 0.4
    bed = None
    estimate = None
    slicer = "unknown"
    profile_name = ""
    unknown = set()
    feature_seen = False
    first_print = None
    first_purge = None
    firmware_retract = 0.8
    firmware_speed = 35.0
    firmware_retracted = False
    line_count = 0
    remaining_s = None
    print_remaining_s = None
    estimate_is_model = False

    def emit(end, de, speed, code):
        nonlocal \
            pos, \
            first_print, \
            first_purge, \
            layer, \
            current_z, \
            pending_layer, \
            print_remaining_s, \
            pending_start, \
            height, \
            last_valid_height
        moving = end != pos
        if not moving and abs(de) < 1e-12:
            pos = end
            return
        if de > 0 and moving and code != 9:
            if (
                pending_layer
                or layer < 0
                or (not layer_explicit and current_z is not None and end[2] > current_z + 1e-4)
            ):
                layer += 1
                z = declared_z if layer_explicit and declared_z is not None else end[2]
                step = z - (current_z or 0)
                if (not height_explicit or height < 0.04) and step >= 0.04:
                    height = step
                if height < 0.04:
                    height = last_valid_height
                last_valid_height = height
                layer_z.append(z)
                current_z = z
                pending_layer = False
                if pending_start is not None:
                    for row_index in range(pending_start, len(rows) // 13):
                        rows[row_index * 13 + 10] = layer
                    pending_start = None
            if first_print is None:
                first_print = len(rows) // 13
                print_remaining_s = remaining_s
        elif de > 0 and moving and code == 9 and first_purge is None:
            first_purge = len(rows) // 13
        a = retract_accel if not moving else travel_accel if de <= 0 else extrusion_accel
        if not math.isfinite(a):
            a = accel
        if moving:
            dx, dy, dz = end[0] - pos[0], end[1] - pos[1], end[2] - pos[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            limit = max(abs(dx) / axis_accel[0], abs(dy) / axis_accel[1], abs(dz) / axis_accel[2])
            if limit:
                a = min(a, distance / limit) if math.isfinite(a) else -distance / limit
            speed_limit = max(
                abs(dx) / axis_speed[0],
                abs(dy) / axis_speed[1],
                abs(dz) / axis_speed[2],
                abs(de) / axis_speed[3],
            )
            if speed_limit:
                speed = min(speed, distance / speed_limit)
        else:
            speed = min(speed, axis_speed[3])
        object_ids.append(current_object if code != 9 else -1)
        cruise_settings.append(cruise_setting if moving else 1.0)
        junctions.append(junction_setting)
        rows.extend(
            (
                *pos,
                *end,
                de,
                width if de > 0 and moving else 0.0,
                height if height >= 0.04 else last_valid_height,
                code if de > 0 else 0,
                layer,
                speed,
                a,
            )
        )
        pos = end

    with path.open("r", encoding="utf-8", errors="replace", buffering=1024 * 1024) as stream:
        for raw in stream:
            line_count += 1
            command, sep, comment = raw.partition(";")
            if sep:
                text = comment.strip()
                low = text.lower()
                if low in ("flush_start", "purge_start", "wipe_start"):
                    if not purge_block:
                        saved_kind = kind
                    purge_block = True
                elif low in ("flush_end", "purge_end", "wipe_end"):
                    purge_block = False
                    if saved_kind is not None:
                        kind = saved_kind
                        saved_kind = None
                label = re.match(r"(start|stop) printing object, unique label id:\s*(\d+)", low)
                if label:
                    current_object = (object_index(("bambu", int(label[2])))
                                      if label[1] == "start" else -1)
                if low.startswith(("feature:", "type:")):
                    feature_seen = True
                    kind = _feature(text.split(":", 1)[1], unknown)
                elif low.startswith("filament_diameter:"):
                    filament = float(_NUMBERS.findall(text.split(":", 1)[1])[0])
                elif low.startswith("nozzle_diameter:"):
                    nozzle = float(_NUMBERS.findall(text.split(":", 1)[1])[0])
                elif low.startswith(("line_width:", "width:")):
                    width = float(text.split(":", 1)[1])
                elif low.startswith(("layer_height:", "height:")):
                    height = float(text.split(":", 1)[1])
                    height_explicit = True
                elif low in ("change_layer", "layer_change") or low.startswith("layer:"):
                    pending_layer = True
                    layer_explicit = True
                    pending_start = len(rows) // 13
                    declared_z = None
                    if kind == 9:
                        kind = 10
                elif low.startswith(("z_height:", "z:")):
                    z = float(text.split(":", 1)[1])
                    declared_z = z
                    if current_z is not None and not pending_layer and abs(z - current_z) > 1e-4:
                        pending_layer = True
                        pending_start = len(rows) // 13
                    if not layer_z:
                        height = z if z > 0 else height
                elif (
                    "estimated printing time" in low
                    or "total estimated time" in low
                    or low.startswith("model printing time:")
                ):
                    value = text.split("=", 1)[-1] if "=" in text else text.split(":", 1)[-1]
                    parsed = _estimate(value)
                    if parsed is not None and (
                        estimate is None or "normal" in low or low.startswith("model printing")
                    ):
                        estimate = parsed
                        estimate_is_model = low.startswith("model printing")
                elif "generated by" in low:
                    slicer = text[low.index("generated by") + len("generated by") :].strip()
                elif "=" in text:
                    key, val = (v.strip() for v in text.split("=", 1))
                    key = key.lower()
                    if key in ("printable_area", "bed_shape"):
                        coords = [float(v) for v in _NUMBERS.findall(val)]
                        if len(coords) >= 6 and len(coords) % 2 == 0:
                            bed = np.array(coords).reshape(-1, 2)
                    elif key in ("printable_size", "bed_size") and bed is None:
                        dims = [float(v) for v in _NUMBERS.findall(val)]
                        if len(dims) >= 2:
                            bed = np.array([[0, 0], dims[:2]])
                    elif key == "filament_diameter":
                        filament = float(_NUMBERS.findall(val)[0])
                    elif key == "nozzle_diameter":
                        nozzle = float(_NUMBERS.findall(val)[0])
                    elif key in ("printer_model", "printer_settings_id"):
                        profile_name = val.strip('"')
                    elif key in (
                        "machine_max_acceleration_x",
                        "machine_max_acceleration_extruding",
                        "default_acceleration",
                    ):
                        value = float(_NUMBERS.findall(val)[0])
                        if value > 0:
                            machine_accel = value
                    elif key in ("retract_length", "retraction_length"):
                        firmware_retract = float(_NUMBERS.findall(val)[0])
                    elif key in ("retract_speed", "retraction_speed"):
                        firmware_speed = float(_NUMBERS.findall(val)[0])
            command = command.strip()
            if not command:
                continue
            # Most sliced motion is whitespace-delimited; regex only for compact G1X1Y2.
            head, _, tail = command.partition(" ")
            if head.startswith("N"):
                command = command.split(None, 1)[-1]
                head, _, tail = command.partition(" ")
            if head == "M486":
                selected = re.search(r"(?:^|\s)S\s*(-?\d+)", tail)
                if selected:
                    source_id = int(selected[1])
                    current_object = object_index(("prusa", source_id)) if source_id >= 0 else -1
                named = re.search(r"(?:^|\s)A(.*)", tail)
                if named and current_object >= 0:
                    objects[current_object]["name"] = named[1].strip().strip('"')
                continue
            if head.startswith("EXCLUDE_OBJECT_"):
                named = re.search(r'NAME=("[^"\n]*"|[^\s]+)', tail)
                if named:
                    name = named[1].strip('"')
                    if head == "EXCLUDE_OBJECT_END":
                        current_object = -1
                    elif head in ("EXCLUDE_OBJECT_DEFINE", "EXCLUDE_OBJECT_START"):
                        index = object_index(("klipper", name), name)
                        if head == "EXCLUDE_OBJECT_START":
                            current_object = index
                        else:
                            center = re.search(r"CENTER=([^\s]+)", tail)
                            polygon = re.search(r"POLYGON=(\[.*\])", tail)
                            if center:
                                objects[index]["center_mm"] = [float(v) for v in center[1].split(",")]
                            if polygon:
                                objects[index]["polygon_mm"] = json.loads(polygon[1])
                elif head == "EXCLUDE_OBJECT_END":
                    current_object = -1
                continue
            # M624 also brackets Bambu layer macros; only unique-label comments
            # identify objects. M625 closes either bracket without creating an id.
            if head == "M625":
                current_object = -1
                continue
            if head not in (
                "G0",
                "G00",
                "G1",
                "G01",
                "G2",
                "G02",
                "G3",
                "G03",
                "G90",
                "G91",
                "G92",
                "G28",
                "G10",
                "G11",
                "G17",
                "G18",
                "G19",
                "M82",
                "M83",
                "M73",
                "M204",
                "M201",
                "M203",
                "M205",
                "M207",
                "SET_VELOCITY_LIMIT",
            ):
                if head.startswith(("G0", "G1", "G2", "G3")) and any(
                    c in head for c in "XYZEFIJRK"
                ):
                    m = re.match(r"G\d+", command)
                    head = m[0]
                    tail = command[len(head) :]
                else:
                    continue
            if head == "G90":
                absolute = True
                continue
            if head == "G91":
                absolute = False
                continue
            if head == "M82":
                eabsolute = True
                continue
            if head == "M83":
                eabsolute = False
                continue
            if head in ("G17", "G18", "G19"):
                plane = head
                continue
            if head == "SET_VELOCITY_LIMIT":
                for token in tail.split():
                    if token.startswith("ACCEL="):
                        accel = float(token.split("=", 1)[1])
                        extrusion_accel = travel_accel = float("nan")
                    elif token.startswith("ACCEL_TO_DECEL="):
                        cruise_setting = -float(token.split("=", 1)[1])
                    elif token.startswith("MINIMUM_CRUISE_RATIO="):
                        cruise_setting = 1.0 - float(token.split("=", 1)[1])
                    elif token.startswith("SQUARE_CORNER_VELOCITY="):
                        junction_setting = -float(token.split("=", 1)[1])
                continue
            words = _words(tail.split("*", 1)[0])
            if head == "M73":
                if "R" in words:
                    remaining_s = words["R"] * 60
                continue
            if head == "G92":
                for k, axis in enumerate("XYZ"):
                    if axis in words:
                        offset[k] = pos[k] - words[axis]
                if "E" in words:
                    epos = words["E"]
                if not words:
                    offset = pos[:]
                    epos = 0.0
                continue
            if head == "G28":
                axes = [k for k, axis in enumerate("XYZ") if axis in tail]
                for k in axes or range(3):
                    pos[k] = 0.0
                    offset[k] = 0.0
                continue
            if head == "M204":
                if words.get("S", 0) > 0:
                    accel = words["S"]
                    extrusion_accel = travel_accel = float("nan")
                if words.get("P", 0) > 0:
                    extrusion_accel = words["P"]
                if words.get("T", 0) > 0:
                    travel_accel = words["T"]
                if words.get("R", 0) > 0:
                    retract_accel = words["R"]
                continue
            if head == "M205":
                if "J" in words and words["J"] >= 0:
                    junction_setting = words["J"]
                continue
            if head == "M203":
                for k, axis in enumerate("XYZE"):
                    if words.get(axis, 0) > 0:
                        axis_speed[k] = words[axis]
                continue
            if head == "M201":
                for k, axis in enumerate("XYZ"):
                    if words.get(axis, 0) > 0:
                        axis_accel[k] = words[axis]
                continue
            if head == "M207":
                firmware_retract = words.get("S", firmware_retract)
                firmware_speed = words.get("F", firmware_speed * 60) / 60
                continue
            if head in ("G10", "G11"):
                # G10 L... is an offset command, not firmware retract.
                if "L" in words or "P" in words:
                    continue
                retract = head == "G10"
                if retract != firmware_retracted:
                    emit(
                        pos[:],
                        -firmware_retract if retract else firmware_retract,
                        firmware_speed,
                        0,
                    )
                    firmware_retracted = retract
                continue
            if "F" in words and words["F"] > 0:
                feed = words["F"] / 60
            end = pos[:]
            for k, axis in enumerate("XYZ"):
                if axis in words:
                    end[k] = words[axis] + offset[k] if absolute else pos[k] + words[axis]
            de = 0.0
            if "E" in words:
                de = words["E"] - epos if eabsolute else words["E"]
                epos = words["E"] if eabsolute else epos + words["E"]
            moving = end != pos
            if de < 0 and moving:
                emit(pos[:], de, feed, 0)
                de = 0.0
            code = 10 if current_object >= 0 and not feature_seen else kind
            if (
                purge_block
                or (print_boundary is not None and line_count < print_boundary)
                or (layer < 0 and not pending_layer and kind == 10 and current_object < 0)
            ):
                code = 9
            if head in ("G2", "G02", "G3", "G03"):
                points = _arc(pos, end, words, head in ("G2", "G02"), plane)
                for point in points:
                    emit(point, de / len(points), feed, code)
            else:
                emit(end, de, feed, code)
    if first_print is None:
        if first_purge is None or layer_explicit or feature_seen:
            raise ValueError(f"No print extrusion found in {path}")
        # Fully unannotated files: the first extrusion is print proper.
        first_print = first_purge
        raw_rows = np.asarray(rows).reshape(-1, 13)
        raw_rows[(raw_rows[:, 6] > 0) & (raw_rows[:, 9] == 9), 9] = 10
        printed = np.flatnonzero(
            (raw_rows[:, 6] > 0) & np.any(raw_rows[:, :3] != raw_rows[:, 3:6], axis=1)
        )
        print_z = raw_rows[printed, 5]
        changed = np.r_[True, np.abs(print_z[1:] - print_z[:-1]) > 1e-4]
        starts = printed[changed]
        raw_rows[:, 10] = np.searchsorted(starts, np.arange(len(raw_rows)), side="right") - 1
        layer_z = print_z[changed].tolist()
    start = min(first_print, first_purge) if first_purge is not None else first_print
    data = np.asarray(rows).reshape(-1, 13)[start:]
    p0 = data[:, :3].astype(np.float32)
    p1 = data[:, 3:6].astype(np.float32)
    e = data[:, 6].astype(np.float32)
    heights = data[:, 8].astype(np.float32)
    widths = data[:, 7].astype(np.float32)
    lengths = np.linalg.norm(p1 - p0, axis=1)
    mask = (e > 0) & (lengths > 1e-9) & (widths <= 0)
    # Rounded rectangle area = h*(w-h)+pi*h²/4.
    widths[mask] = e[mask] * (math.pi * filament**2 / 4) / (
        lengths[mask] * heights[mask]
    ) + heights[mask] * (1 - math.pi / 4)
    acceleration = np.where(np.isfinite(data[:, 12]), data[:, 12], machine_accel)
    acceleration = np.where(
        acceleration < 0, np.minimum(machine_accel, -acceleration), acceleration
    )
    junction_values = np.asarray(junctions)[start:]
    jd = np.where(
        junction_values < 0,
        junction_values**2 * (math.sqrt(2) - 1) / np.maximum(acceleration, 1e-6),
        junction_values,
    )
    cruise_values = np.asarray(cruise_settings)[start:]
    cruise_ratio = np.where(
        cruise_values < 0, -cruise_values / np.maximum(acceleration, 1e-6), cruise_values
    )
    duration = _durations(
        p0, p1, e, data[:, 11], np.maximum(acceleration, 1e-6), jd, np.clip(cruise_ratio, 1e-3, 1.0)
    )
    t1 = np.cumsum(duration)
    t0 = np.r_[0.0, t1[:-1]]
    origin = t0[first_print - start]
    t0 -= origin
    t1 -= origin
    kinds = data[:, 9].astype(np.uint8)
    layers = data[:, 10].astype(np.int32)
    lz = np.asarray(layer_z, dtype=np.float32)
    lt = np.empty(len(lz), dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, layers[1:] != layers[:-1]])
    valid = layers[starts] >= 0
    lt[layers[starts[valid]]] = t0[starts[valid]]
    obj_ids = np.asarray(object_ids, dtype=np.int16)[start:]
    for record in objects:
        assigned = obj_ids == record["id"]
        record["n_moves"] = int(np.count_nonzero(assigned))
        extruded = assigned & (e > 0) & (lengths > 0)
        object_points = np.concatenate((p0[extruded], p1[extruded]))
        if len(object_points):
            lo, hi = object_points.min(0), object_points.max(0)
            record["bbox_mm"] = [lo.tolist(), hi.tolist()]
            if record["center_mm"] is None:
                record["center_mm"] = ((lo[:2] + hi[:2]) / 2).tolist()
    obj = (e > 0) & (lengths > 0) & (
        (obj_ids >= 0) if objects else np.isin(kinds, range(1, 9))
    )
    points = np.concatenate((p0[obj], p1[obj]))
    if not len(points):
        fallback = (e > 0) & (lengths > 0) & (kinds != 9)
        points = np.concatenate((p0[fallback], p1[fallback]))
    if bed is None:
        LOG.warning("No printable_area/bed_shape in %s; bed size and centre unknown", path)
    name = profile_name.lower()
    if printer_profile is None:
        printer_profile = next(
            (
                code
                for part, code in [
                    ("x1", "bambu_x1c"),
                    ("mk4", "prusa_mk4"),
                    ("voron", "voron24_300"),
                    ("a1 mini", "bambu_a1mini"),
                ]
                if part in name
            ),
            "unknown",
        )
        sidecar = path.with_name("slice.json")
        if sidecar.exists():
            info = json.loads(sidecar.read_text())
            printer_profile = info.get("printer_profile", printer_profile)
    comparable_estimate = (
        estimate
        if estimate_is_model or print_remaining_s is None
        else min(estimate or print_remaining_s, print_remaining_s)
    )
    meta = {
        "printer_profile": printer_profile,
        "bed_size_mm": (bed.max(0) - bed.min(0)).tolist() if bed is not None else None,
        "plate_center_mm": ((bed.max(0) + bed.min(0)) / 2).tolist() if bed is not None else None,
        "nozzle_diameter": nozzle,
        "filament_diameter": filament,
        "total_time_s": float(t1[-1]),
        "total_e_mm": float(np.sum(e[e > 0], dtype=np.float64)),
        "objects": objects,
        "object_bbox_mm": [points.min(0).tolist(), points.max(0).tolist()],
        "slicer": slicer,
        "source_gcode": str(path.resolve()),
        "slicer_estimated_time_s": estimate,
        "time_ratio": float(t1[-1] / estimate) if estimate else None,
        "slicer_print_estimated_time_s": comparable_estimate,
        "print_time_ratio": float(t1[-1] / comparable_estimate) if comparable_estimate else None,
        "source_lines": line_count,
        "time_origin": "first print extrusion; startup purge has negative times",
    }
    return Timeline(
        {
            "p0": p0,
            "p1": p1,
            "t0": t0,
            "t1": t1,
            "e": e,
            "width": widths,
            "height": heights,
            "kind": kinds,
            "obj": obj_ids,
            "layer": layers,
            "layer_z": lz,
            "layer_t0": lt,
            "meta": meta,
        }
    )


def save_timeline(timeline, path):
    """Atomically write a compressed archive; safe to resume after interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".timeline-", suffix=".npz", delete=False
        ) as out:
            tmp = Path(out.name)
            arrays = {
                k: getattr(timeline, k)
                for k in (
                    "p0",
                    "p1",
                    "t0",
                    "t1",
                    "e",
                    "width",
                    "height",
                    "kind",
                    "obj",
                    "layer",
                    "layer_z",
                    "layer_t0",
                )
            }
            np.savez_compressed(out, **arrays, meta=np.array(json.dumps(timeline.meta)))
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def summary(timeline):
    meta = timeline.meta
    return {
        "moves": len(timeline.e),
        "layers": len(timeline.layer_z),
        "total_time_s": meta["total_time_s"],
        "slicer_estimated_time_s": meta.get("slicer_estimated_time_s"),
        "time_ratio": meta.get("time_ratio"),
        "slicer_print_estimated_time_s": meta.get("slicer_print_estimated_time_s"),
        "print_time_ratio": meta.get("print_time_ratio"),
        "kinds": {
            KINDS[int(k)]: int(n)
            for k, n in zip(*np.unique(timeline.kind, return_counts=True), strict=False)
        },
        "bbox_mm": meta["object_bbox_mm"],
    }


def add_arguments(parser):
    parser.add_argument("gcode", nargs="?", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Resume missing timelines under SCYTHE_GCODE_CACHE (or $ROOT/gcode_cache)",
    )
    parser.add_argument("--root", type=Path, help="Data root (defaults to SCYTHE_SYNTH_ROOT)")
    parser.add_argument("--force", action="store_true", help="Regenerate existing timelines")
    parser.add_argument("--printer-profile")
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel file workers for --all (default: 1)"
    )


def _convert_file(path, output, printer_profile):
    before = time.perf_counter()
    timeline = parse_gcode(path, printer_profile=printer_profile)
    parsed = time.perf_counter()
    save_timeline(timeline, output)
    return dict(
        source=str(path),
        output=str(output),
        parse_seconds=parsed - before,
        write_seconds=time.perf_counter() - parsed,
        **summary(timeline),
    )


def run(args):
    from .paths import GCODE_CACHE

    workers = getattr(args, "workers", 1)
    if workers < 1:
        raise ValueError("--workers must be positive")
    if args.all:
        if args.gcode or args.output:
            raise ValueError("--all cannot be combined with a file or output")
        cache = args.root / "gcode_cache" if args.root is not None else GCODE_CACHE
        paths = sorted(cache.rglob("model.gcode"))
    elif args.gcode:
        paths = [args.gcode]
    else:
        raise ValueError("Specify a G-code file or --all")
    jobs = [(path, args.output or path.with_name("timeline.npz")) for path in paths]
    jobs = [(path, output) for path, output in jobs if not (args.all and output.exists() and not getattr(args, "force", False))]
    failures = 0
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context

        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
            pending = {
                executor.submit(_convert_file, path, output, args.printer_profile): path
                for path, output in jobs
            }
            for future in as_completed(pending):
                try:
                    print(json.dumps(future.result()), flush=True)
                except (OSError, ValueError) as exc:
                    LOG.error("%s: %s", pending[future], exc)
                    failures += 1
    else:
        for path, output in jobs:
            try:
                print(json.dumps(_convert_file(path, output, args.printer_profile)), flush=True)
            except (OSError, ValueError) as exc:
                LOG.error("%s: %s", path, exc)
                failures += 1
    return int(failures > 0)


# Kept identical to the standalone NumPy-only Blender reference reader.
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
        return self.p0[i].astype(np.float64) + (self.p1[i].astype(np.float64) - self.p0[i]) * f, i

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
        if self.t0[i] <= t < self.t1[i] and self.e[i] > 0 and np.any(self.p0[i] != self.p1[i]):
            return i, f
        return None

    def layer_range(self, layer):
        indices = np.flatnonzero(self.layer == layer)
        return slice(int(indices[0]), int(indices[-1]) + 1) if len(indices) else slice(0, 0)

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
