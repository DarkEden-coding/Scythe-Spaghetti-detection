"""Scene plumbing and geometry primitives shared by every setup.

Units are metres, Z up. Helpers create an object, link it into a collection,
give it one material and a pass index, and optionally parent it. Parenting
never sets a parent-inverse, so coordinates are always parent-local.
"""

import math

import bmesh
import bpy
from mathutils import Vector
from mathutils import noise as mnoise

PASS_ENVIRONMENT = 1
PASS_PRINTER = 2
PASS_PRINT = 3
PASS_SPAGHETTI = 4
PASS_DETACHED = 5
PASS_DELAMINATION = PASS_DETACHED
PASS_SHIFTED = 6
PASS_NOZZLE_BLOB = 7
PASS_OCCLUDER = 8

COLLECTIONS = ("Environment", "Printer", "Print", "Failure", "CameraRig", "Lights")

LAYER = 0.0002
LINE_W = 0.00045
FPS = 30

_NV_OFF = (Vector((31.7, 17.3, 5.9)), Vector((-13.1, 47.9, 23.3)))


def noise_vector(p):
    """Vector Perlin noise. mathutils.noise.noise_vector uses a random table
    that differs between Blender processes (noise.seed_set does not fix it),
    so this is three scalar noise lookups, which are reproducible."""
    return Vector((mnoise.noise(p), mnoise.noise(p + _NV_OFF[0]), mnoise.noise(p + _NV_OFF[1])))


def clear_scene():
    """Remove every object and every datablock a scene build can create, so a
    persistent worker starts each job from the same empty state."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.curves, bpy.data.materials,
                  bpy.data.lights, bpy.data.cameras, bpy.data.node_groups,
                  bpy.data.actions, bpy.data.images, bpy.data.textures,
                  bpy.data.collections):
        if len(block):
            bpy.data.batch_remove(list(block))
    for world in list(bpy.data.worlds):
        if world.users == 0:
            bpy.data.worlds.remove(world)
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def make_collections():
    root = bpy.context.scene.collection
    colls = {}
    for name in COLLECTIONS:
        c = bpy.data.collections.new(name)
        root.children.link(c)
        colls[name] = c
    return colls


def link(obj, coll, mat=None, pass_index=PASS_PRINTER, parent=None):
    coll.objects.link(obj)
    if mat is not None:
        obj.data.materials.append(mat)
    obj.pass_index = pass_index
    obj.parent = parent
    return obj


def empty(name, coll, location=(0, 0, 0), parent=None):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_size = 0.02
    obj.location = location
    coll.objects.link(obj)
    obj.parent = parent
    return obj


def aim(obj, target, roll=0.0):
    """Point -Z of `obj` at `target` (both in the object's parent space)."""
    direction = Vector(target) - obj.location
    rot = direction.to_track_quat("-Z", "Y").to_euler()
    obj.rotation_euler = rot
    if roll:
        obj.rotation_euler.rotate_axis("Z", roll)


def _cuboid(bm, center, size):
    geom = bmesh.ops.create_cube(bm, size=1.0)
    for v in geom["verts"]:
        v.co = Vector((center[0] + v.co.x * size[0], center[1] + v.co.y * size[1],
                       center[2] + v.co.z * size[2]))
    return geom["verts"]


def box(name, center, size, mat, coll, pass_index=PASS_PRINTER, bevel=0.0,
        parent=None, rotation=(0, 0, 0)):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    _cuboid(bm, (0, 0, 0), size)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = center
    obj.rotation_euler = rotation
    if bevel > 0:
        mod = obj.modifiers.new("Bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 3
        mod.limit_method = "ANGLE"
    return link(obj, coll, mat, pass_index, parent)


AXIS_ROT = {"Z": (0, 0, 0), "X": (0, math.pi / 2, 0), "Y": (math.pi / 2, 0, 0)}


def cylinder(name, center, radius, depth, mat, coll, axis="Z", verts=32,
             pass_index=PASS_PRINTER, parent=None, cone_r2=None, bevel=0.0):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=verts, radius1=radius,
                          radius2=radius if cone_r2 is None else cone_r2,
                          depth=depth)
    bm.to_mesh(mesh)
    bm.free()
    mesh.shade_smooth()
    mesh.set_sharp_from_angle(angle=math.radians(40))
    obj = bpy.data.objects.new(name, mesh)
    obj.location = center
    obj.rotation_euler = AXIS_ROT[axis]
    if bevel > 0:
        mod = obj.modifiers.new("Bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 2
        mod.limit_method = "ANGLE"
    return link(obj, coll, mat, pass_index, parent)


def tube(name, splines, radius, mat, coll, cyclic=False, pass_index=PASS_PRINTER,
         parent=None, resolution=3, z_squash=1.0, origin=(0, 0, 0)):
    """Poly-curve tube through each point list in `splines`. `z_squash` flattens
    the cross-section about `origin` (used for extruded lines)."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = resolution
    cu.use_fill_caps = True
    for pts in splines:
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        for i, p in enumerate(pts):
            sp.points[i].co = (p[0] - origin[0], p[1] - origin[1],
                               (p[2] - origin[2]) / z_squash, 1.0)
        sp.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, cu)
    obj.location = origin
    obj.scale = (1, 1, z_squash)
    return link(obj, coll, mat, pass_index, parent)


def ring(name, center, major, minor, mat, coll, axis="Z", pass_index=PASS_PRINTER,
         parent=None):
    pts = [(major * math.cos(a), major * math.sin(a), 0)
           for a in (2 * math.pi * i / 48 for i in range(48))]
    obj = tube(name, [pts], minor, mat, coll, cyclic=True, pass_index=pass_index,
               parent=parent, resolution=2)
    obj.location = center
    obj.rotation_euler = AXIS_ROT[axis]
    return obj


def rounded_outline(hx, hy, radius, n=8):
    pts = []
    for cx, cy, a0 in ((hx - radius, hy - radius, 0), (-hx + radius, hy - radius, 90),
                       (-hx + radius, -hy + radius, 180), (hx - radius, -hy + radius, 270)):
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return pts


def filled_profile(name, loops, thickness, center, mat, coll, pass_index=PASS_PRINTER,
                   parent=None):
    """Even-odd filled 2D curve extruded to `thickness`, centred on `center`.
    `loops` are lists of (x, y[, z]) points; inner loops become holes."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "2D"
    cu.fill_mode = "BOTH"
    cu.extrude = thickness / 2
    for pts in loops:
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        for i, p in enumerate(pts):
            sp.points[i].co = (p[0], p[1], 0, 1)
        sp.use_cyclic_u = True
    obj = bpy.data.objects.new(name, cu)
    obj.location = center
    return link(obj, coll, mat, pass_index, parent)


def rounded_slab(name, center, half, radius, thickness, mat, coll,
                 pass_index=PASS_PRINTER, parent=None):
    """Flat plate with rounded vertical corners."""
    return filled_profile(name, [rounded_outline(half[0], half[1], radius)], thickness,
                          center, mat, coll, pass_index, parent)


def extrusion(name, center, length, axis, mat, slot_mat, coll, profile=(0.02, 0.02),
              pass_index=PASS_PRINTER, parent=None):
    """Aluminium V-slot extrusion: a bar plus dark slot strips every 20 mm on
    each long face. `profile` is the cross-section (first, second cross axis)."""
    a = "XYZ".index(axis)
    b, c = [i for i in range(3) if i != a]
    size = [0.0, 0.0, 0.0]
    size[a], size[b], size[c] = length, profile[0], profile[1]

    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    _cuboid(bm, (0, 0, 0), size)
    core_faces = len(bm.faces)
    for normal_axis, along_axis in ((b, c), (c, b)):
        width = size[along_axis]
        for i in range(max(1, round(width / 0.02))):
            pos = -width / 2 + 0.01 + 0.02 * i
            for sign in (-1, 1):
                ctr = [0.0, 0.0, 0.0]
                ctr[normal_axis] = sign * size[normal_axis] / 2
                ctr[along_axis] = pos
                s = [0.0, 0.0, 0.0]
                s[a], s[normal_axis], s[along_axis] = length * 0.998, 0.0008, 0.0062
                _cuboid(bm, ctr, s)
    for i, f in enumerate(bm.faces):
        f.material_index = 0 if i < core_faces else 1
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = center
    link(obj, coll, mat, pass_index, parent)
    obj.data.materials.append(slot_mat)
    return obj


def fan(name, center, radius, axis, m, coll, parent=None, pass_index=PASS_PRINTER,
        rings=4):
    """Fan opening: dark void disc, concentric guard rings and a hub."""
    normal = Vector({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[axis.lstrip("-")])
    if axis.startswith("-"):
        normal = -normal
    ctr = Vector(center)
    cylinder(f"{name}Void", ctr - normal * 0.0006, radius * 1.05, 0.001, m["void"], coll,
             axis=axis.lstrip("-"), parent=parent, pass_index=pass_index)
    for i in range(rings):
        r = radius * (i + 1) / rings
        ring(f"{name}Ring{i}", ctr, r, max(0.0006, radius * 0.045), m["sheet"], coll,
             axis=axis.lstrip("-"), parent=parent, pass_index=pass_index)
    cylinder(f"{name}Hub", ctr, radius * 0.32, 0.0016, m["fan_hub"], coll,
             axis=axis.lstrip("-"), parent=parent, pass_index=pass_index)


def catmull(points, per_seg=12):
    """Sample a Catmull-Rom spline through `points` (list of 3-tuples)."""
    pts = [Vector(p) for p in points]
    pts = [pts[0]] + pts + [pts[-1]]
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for s in range(per_seg):
            t = s / per_seg
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(pts[-2])
    return [tuple(p) for p in out]


def area_light(name, coll, location, target, energy, color, size, size_y=None,
               shape="RECTANGLE", spread=None, parent=None):
    light = bpy.data.lights.new(name, "AREA")
    light.shape = shape
    light.size = size
    if size_y is not None:
        light.size_y = size_y
    light.energy = energy
    light.color = color
    if spread is not None:
        light.spread = spread
    obj = bpy.data.objects.new(name, light)
    obj.location = location
    coll.objects.link(obj)
    obj.parent = parent
    aim(obj, target)
    return obj


def point_light(name, coll, location, energy, color, radius=0.002, parent=None,
                spot=None, target=None):
    light = bpy.data.lights.new(name, "SPOT" if spot else "POINT")
    light.energy = energy
    light.color = color
    light.shadow_soft_size = radius
    if spot:
        light.spot_size = spot
        light.spot_blend = 0.6
    obj = bpy.data.objects.new(name, light)
    obj.location = location
    coll.objects.link(obj)
    obj.parent = parent
    if target is not None:
        aim(obj, target)
    return obj


def set_world(color, strength):
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    if world.node_tree is None:
        world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*color, 1)
    bg.inputs["Strength"].default_value = strength
