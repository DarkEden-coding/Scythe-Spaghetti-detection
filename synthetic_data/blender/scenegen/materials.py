"""Material factory.

Every setup asks for the same roles (frame, accent, plate, filament, ...)
through a palette, so printer and environment builders never hard-code
colours. Palette keys:

    frame      "black" | "silver"   aluminium extrusions / sheet metal
    accent     rgb                  printed parts (Prusa orange, Voron red...)
    body       rgb                  toolhead body
    cover      rgb                  toolhead front cover
    plate      PLATES key           build surface
    filament   rgb                  what is being printed
    spool      rgb                  spare spool on the desk (optional)
    infrared   bool                 scene is seen by an IR-only camera (optional)
    layer      float                layer height in metres for layer lines (optional)
"""

import bpy

from .util import LAYER


def new_material(name):
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:
        mat.use_nodes = True
    return mat


def principled(name, color, roughness=0.5, metallic=0.0, **extra):
    mat = new_material(name)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    for key, value in extra.items():
        bsdf.inputs[key].default_value = value
    return mat


def emission(name, color, strength, base=(0.02, 0.02, 0.02)):
    return principled(name, base, 0.4, **{"Emission Color": (*color, 1.0),
                                          "Emission Strength": strength})


def add_bump(mat, height_socket, distance, strength=1.0):
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = distance
    links.new(height_socket, bump.inputs["Height"])
    links.new(bump.outputs["Normal"], nodes["Principled BSDF"].inputs["Normal"])
    return bump


def noise(mat, scale, detail=4.0, roughness=0.6, coord="Object"):
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    tc = nodes.get("TexCoord") or nodes.new("ShaderNodeTexCoord")
    tc.name = "TexCoord"
    tex = nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = scale
    tex.inputs["Detail"].default_value = detail
    tex.inputs["Roughness"].default_value = roughness
    links.new(tc.outputs[coord], tex.inputs["Vector"])
    return tex


def strand_texture(mat, color):
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    ao = nodes.new("ShaderNodeAmbientOcclusion")
    ao.only_local = True
    ao.samples = 4
    ao.inputs["Distance"].default_value = 0.0025
    speck = noise(mat, 650.0, detail=1.0, roughness=0.5)
    ramp = nodes.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = 0.3
    ramp.inputs["From Max"].default_value = 0.7
    ramp.inputs["To Min"].default_value = 0.45
    ramp.inputs["To Max"].default_value = 1.2
    links.new(speck.outputs["Fac"], ramp.inputs["Value"])
    shade = nodes.new("ShaderNodeMapRange")
    shade.inputs["To Min"].default_value = 0.08
    links.new(ao.outputs["AO"], shade.inputs["Value"])
    mul = nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    links.new(ramp.outputs["Result"], mul.inputs[0])
    links.new(shade.outputs["Result"], mul.inputs[1])
    tint = nodes.new("ShaderNodeMix")
    tint.data_type = "RGBA"
    tint.blend_type = "MULTIPLY"
    tint.inputs["Factor"].default_value = 1.0
    tint.inputs["A"].default_value = (*color, 1.0)
    links.new(mul.outputs["Value"], tint.inputs["B"])
    links.new(tint.outputs["Result"], nodes["Principled BSDF"].inputs["Base Color"])


def vary_roughness(mat, scale, lo, hi):
    """Smudges / wear: drive roughness from low-frequency noise."""
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    tex = noise(mat, scale, detail=6.0, coord="Generated")
    rng = nodes.new("ShaderNodeMapRange")
    rng.inputs["From Min"].default_value = 0.3
    rng.inputs["From Max"].default_value = 0.7
    rng.inputs["To Min"].default_value = lo
    rng.inputs["To Max"].default_value = hi
    links.new(tex.outputs["Fac"], rng.inputs["Value"])
    links.new(rng.outputs["Result"], nodes["Principled BSDF"].inputs["Roughness"])


def color_variation(mat, scale, lo, hi, coord="Generated"):
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    tex = noise(mat, scale, coord=coord)
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*lo, 1)
    ramp.color_ramp.elements[1].color = (*hi, 1)
    links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    return tex


def layer_line_bump(mat, strength, layer=LAYER):
    """Rounded FDM layer ridges: sqrt(1 - (2*fract(z/L) - 1)^2) in object Z,
    with a per-layer random amplitude so walls are not perfectly regular.
    Object space, so the ridges follow the bed (and a tilted delaminated
    section) rather than world Z."""
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    tc = nodes.get("TexCoord") or nodes.new("ShaderNodeTexCoord")
    tc.name = "TexCoord"
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(tc.outputs["Object"], sep.inputs["Vector"])

    def math_node(op, a, b=None, c=None):
        n = nodes.new("ShaderNodeMath")
        n.operation = op
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, bpy.types.NodeSocket):
                links.new(v, n.inputs[i])
            else:
                n.inputs[i].default_value = v
        return n.outputs[0]

    layer = math_node("DIVIDE", sep.outputs["Z"], layer)
    centred = math_node("MULTIPLY_ADD", math_node("FRACT", layer), 2.0, -1.0)
    ridge = math_node("SQRT", math_node("SUBTRACT", 1.0,
                                        math_node("MULTIPLY", centred, centred)))
    wn = nodes.new("ShaderNodeTexWhiteNoise")
    wn.noise_dimensions = "1D"
    links.new(math_node("FLOOR", layer), wn.inputs["W"])
    jitter = math_node("MULTIPLY_ADD", wn.outputs["Value"], 0.35, 0.65)
    add_bump(mat, math_node("MULTIPLY", ridge, jitter), 0.00006, strength)


def add_dust(mat, amount, color=(0.32, 0.31, 0.29)):
    """Dust settling on upward-facing surfaces, in patchy noise."""
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    base = bsdf.inputs["Base Color"]
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    if base.is_linked:
        links.new(base.links[0].from_socket, mix.inputs["A"])
    else:
        mix.inputs["A"].default_value = base.default_value
    mix.inputs["B"].default_value = (*color, 1)
    geo = nodes.new("ShaderNodeNewGeometry")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(geo.outputs["Normal"], sep.inputs["Vector"])
    up = nodes.new("ShaderNodeMapRange")
    up.inputs["From Min"].default_value = 0.4
    up.inputs["From Max"].default_value = 1.0
    links.new(sep.outputs["Z"], up.inputs["Value"])
    patch = noise(mat, 25.0, detail=8.0, coord="Generated")
    patches = nodes.new("ShaderNodeMapRange")
    patches.inputs["From Min"].default_value = 0.4
    patches.inputs["From Max"].default_value = 0.75
    patches.inputs["To Max"].default_value = amount
    links.new(patch.outputs["Fac"], patches.inputs["Value"])
    fac = nodes.new("ShaderNodeMath")
    fac.operation = "MULTIPLY"
    links.new(up.outputs["Result"], fac.inputs[0])
    links.new(patches.outputs["Result"], fac.inputs[1])
    links.new(fac.outputs[0], mix.inputs["Factor"])
    links.new(mix.outputs["Result"], base)


def textured_pei(name, lo, hi, metallic, rough=(0.38, 0.62)):
    """Powder-coated PEI: voronoi grit + fine grain, low-frequency tint and
    fingerprint/glue-residue roughness variation."""
    mat = principled(name, hi, 0.5, metallic=metallic)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    tc = nodes.get("TexCoord") or nodes.new("ShaderNodeTexCoord")
    tc.name = "TexCoord"
    vor = nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 2600.0
    links.new(tc.outputs["Object"], vor.inputs["Vector"])
    grain = noise(mat, 6000.0, detail=2.0)
    mix = nodes.new("ShaderNodeMath")
    mix.operation = "ADD"
    links.new(vor.outputs["Distance"], mix.inputs[0])
    links.new(grain.outputs["Fac"], mix.inputs[1])
    add_bump(mat, mix.outputs[0], 0.00012, 0.6)
    color_variation(mat, 12.0, lo, hi)
    vary_roughness(mat, 7.0, *rough)
    return mat


def thin_panel(name, tint, roughness=0.02, scuff=0.05):
    """Thin glass/acrylic sheet: tinted see-through plus a Fresnel reflection.

    A transparent BSDF (rather than refraction) lets shadows and the object
    index pass go straight through, so masks show what is behind a door or
    panel, and a 3 mm sheet barely displaces the view anyway."""
    mat = new_material(name)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.remove(nodes["Principled BSDF"])
    see = nodes.new("ShaderNodeBsdfTransparent")
    see.inputs["Color"].default_value = (*tint, 1)
    gloss = nodes.new("ShaderNodeBsdfGlossy")
    gloss.inputs["Color"].default_value = (1, 1, 1, 1)
    fres = nodes.new("ShaderNodeFresnel")
    fres.inputs["IOR"].default_value = 1.5
    mix = nodes.new("ShaderNodeMixShader")
    links.new(fres.outputs[0], mix.inputs["Fac"])
    links.new(see.outputs[0], mix.inputs[1])
    links.new(gloss.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], nodes["Material Output"].inputs["Surface"])
    tex = noise(mat, 6.0, detail=6.0, coord="Generated")
    rng = nodes.new("ShaderNodeMapRange")
    rng.inputs["From Min"].default_value = 0.3
    rng.inputs["From Max"].default_value = 0.7
    rng.inputs["To Min"].default_value = roughness
    rng.inputs["To Max"].default_value = roughness + scuff
    links.new(tex.outputs["Fac"], rng.inputs["Value"])
    links.new(rng.outputs["Result"], gloss.inputs["Roughness"])
    return mat


def plate_material(kind):
    if kind == "textured_gold":
        return textured_pei("TexturedPEI", (0.30, 0.19, 0.06), (0.42, 0.28, 0.10), 0.35)
    if kind == "textured_black":
        return textured_pei("TexturedPEIBlack", (0.018, 0.018, 0.02), (0.035, 0.034, 0.036),
                            0.25, rough=(0.45, 0.7))
    if kind == "smooth_pei":
        mat = principled("SmoothPEI", (0.55, 0.33, 0.06), 0.18)
        color_variation(mat, 6.0, (0.48, 0.28, 0.05), (0.60, 0.37, 0.08))
        vary_roughness(mat, 9.0, 0.08, 0.35)  # wiped / greasy patches
        return mat
    if kind == "satin_black":
        mat = principled("SatinSheet", (0.035, 0.035, 0.038), 0.42, metallic=0.2)
        add_bump(mat, noise(mat, 4000.0, detail=2.0).outputs["Fac"], 0.00005, 0.3)
        vary_roughness(mat, 8.0, 0.32, 0.55)
        return mat
    if kind == "glass":
        mat = principled("BedGlass", (0.9, 0.93, 0.9), 0.02,
                         **{"Transmission Weight": 1.0, "IOR": 1.5})
        vary_roughness(mat, 10.0, 0.0, 0.12)  # glue stick / fingerprints
        return mat
    raise ValueError(kind)


def ir_albedo(rgb):
    """Near-IR reflectance of a dyed plastic. Most dyes are transparent at
    850 nm, so coloured filament looks bright under an IR illuminator; only
    carbon-black-loaded ones stay dark."""
    if max(rgb) < 0.15:
        return rgb
    v = 0.55 + 0.2 * max(rgb)
    return (v, v, v)


def build_materials(palette):
    m = {}
    if palette.get("infrared"):
        palette = dict(palette, filament=ir_albedo(palette["filament"]))
    m["env_dark"] = principled("ChamberPlastic", (0.03, 0.03, 0.033), 0.5)
    add_bump(m["env_dark"], noise(m["env_dark"], 900.0, detail=2.0).outputs["Fac"], 0.0002, 0.25)
    vary_roughness(m["env_dark"], 4.0, 0.42, 0.7)
    m["sheet"] = principled("PowderCoat", (0.022, 0.022, 0.024), 0.42)
    add_bump(m["sheet"], noise(m["sheet"], 1500.0).outputs["Fac"], 0.0001, 0.2)

    if palette["frame"] == "silver":
        m["frame"] = principled("FrameAlu", (0.62, 0.62, 0.63), 0.38, metallic=1.0)
    else:
        m["frame"] = principled("FrameAlu", (0.025, 0.025, 0.028), 0.4, metallic=0.75)
    vary_roughness(m["frame"], 15.0, 0.3, 0.5)

    m["accent"] = principled("PrintedParts", palette["accent"], 0.45)
    add_bump(m["accent"], noise(m["accent"], 2500.0, detail=2.0).outputs["Fac"], 0.00008, 0.3)
    m["body"] = principled("ToolheadBody", palette.get("body", (0.028, 0.028, 0.03)), 0.38)
    vary_roughness(m["body"], 25.0, 0.3, 0.5)
    m["cover"] = principled("ToolheadCover", palette.get("cover", (0.09, 0.09, 0.095)), 0.33)
    vary_roughness(m["cover"], 25.0, 0.25, 0.45)

    m["plate"] = plate_material(palette["plate"])
    m["plate_edge"] = principled("PlateSteelEdge", (0.03, 0.03, 0.03), 0.35, metallic=0.9)
    m["anodized"] = principled("BlackAnodized", (0.03, 0.03, 0.035), 0.35, metallic=0.8)
    m["pcb"] = principled("HeatbedPCB", (0.012, 0.014, 0.013), 0.3)
    m["silicone_heater"] = principled("SiliconeHeater", (0.45, 0.04, 0.03), 0.6)
    m["steel"] = principled("ChromeRod", (0.85, 0.85, 0.85), 0.14, metallic=1.0)
    vary_roughness(m["steel"], 20.0, 0.08, 0.3)

    m["screw"] = principled("LeadScrew", (0.75, 0.75, 0.74), 0.25, metallic=1.0)
    nodes, links = m["screw"].node_tree.nodes, m["screw"].node_tree.links
    tc = nodes.new("ShaderNodeTexCoord")
    wave = nodes.new("ShaderNodeTexWave")
    wave.bands_direction = "Z"
    wave.inputs["Scale"].default_value = 125.0  # ~2 mm pitch in object space
    links.new(tc.outputs["Object"], wave.inputs["Vector"])
    add_bump(m["screw"], wave.outputs["Fac"], 0.0004, 1.0)

    m["carbon"] = principled("CarbonRod", (0.018, 0.018, 0.02), 0.28)
    m["aluminium"] = principled("HeaterBlock", (0.75, 0.74, 0.72), 0.35, metallic=1.0)
    m["sock"] = principled("SiliconeSock", (0.32, 0.33, 0.35), 0.75)
    m["brass"] = principled("Nozzle", (0.85, 0.58, 0.28), 0.3, metallic=1.0)
    m["rubber"] = principled("Belt", (0.014, 0.014, 0.014), 0.75)
    m["motor"] = principled("Stepper", (0.02, 0.02, 0.022), 0.5)
    m["spring"] = principled("Spring", (0.75, 0.55, 0.02), 0.35, metallic=0.6)
    m["ptfe"] = principled("PTFE", (0.85, 0.86, 0.86), 0.35, **{"Transmission Weight": 0.45})
    m["glass"] = thin_panel("Glass", (0.86, 0.9, 0.87), 0.0, 0.08)
    m["acrylic_smoke"] = thin_panel("SmokedAcrylic", (0.3, 0.3, 0.31), 0.01, 0.025)
    m["acrylic"] = thin_panel("ClearAcrylic", (0.9, 0.91, 0.91), 0.01, 0.025)  # scuffs
    m["void"] = principled("Void", (0.002, 0.002, 0.002), 0.9)
    m["fan_hub"] = principled("FanHub", (0.05, 0.05, 0.055), 0.6)
    m["cable"] = principled("CableSleeve", (0.02, 0.02, 0.022), 0.7)

    # PLA: glossy-ish with a little subsurface so thin walls glow at edges.
    sss = {"Subsurface Weight": 0.25, "Subsurface Radius": (1.0, 0.6, 0.4),
           "Subsurface Scale": 0.0015}
    m["pla"] = principled("Filament", palette["filament"], 0.3, **sss)
    layer_line_bump(m["pla"], 0.9, palette.get("layer", LAYER))
    m["pla_top"] = principled("FilamentFresh", palette["filament"], 0.25, **sss)
    # Failure strands: the nest's self-shadowing gaps are baked into the base
    # colour (local AO plus per-strand speckle) so they reach the albedo pass
    # and survive the denoiser; lit only, they melt into a smooth skin.
    m["strand"] = principled("FilamentStrand", palette["filament"], 0.25, **sss)
    strand_texture(m["strand"], palette["filament"])
    # Cut face of the STL body under the live toolpath layers: what shows
    # through sparse infill is mostly shadowed lower infill.
    m["cap"] = principled("FilamentDeep", tuple(c * 0.4 for c in palette["filament"]), 0.6)

    # Room materials.
    m["wood"] = principled("DeskWood", (0.35, 0.21, 0.11), 0.45)
    nodes, links = m["wood"].node_tree.nodes, m["wood"].node_tree.links
    tc = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (1.0, 14.0, 1.0)
    wave = nodes.new("ShaderNodeTexWave")
    wave.inputs["Scale"].default_value = 3.0
    wave.inputs["Distortion"].default_value = 6.0
    wave.inputs["Detail"].default_value = 3.0
    links.new(tc.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.22, 0.12, 0.055, 1)
    ramp.color_ramp.elements[1].color = (0.42, 0.26, 0.14, 1)
    links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    vary_roughness(m["wood"], 5.0, 0.3, 0.6)

    m["paint"] = principled("WallPaint", palette.get("wall", (0.55, 0.56, 0.52)), 0.85)
    add_bump(m["paint"], noise(m["paint"], 300.0).outputs["Fac"], 0.0004, 0.2)
    m["lack"] = principled("LackWhite", (0.78, 0.78, 0.76), 0.6)
    vary_roughness(m["lack"], 6.0, 0.5, 0.7)
    m["cardboard"] = principled("Cardboard", (0.42, 0.29, 0.16), 0.8)
    color_variation(m["cardboard"], 8.0, (0.36, 0.24, 0.13), (0.46, 0.32, 0.18))
    m["ceramic"] = principled("Mug", palette.get("mug", (0.7, 0.72, 0.74)), 0.12)
    m["spool"] = principled("SpoolFilament", palette.get("spool", (0.6, 0.6, 0.6)), 0.3)
    m["spool_flange"] = principled("SpoolFlange", (0.02, 0.02, 0.02), 0.35)
    m["concrete"] = principled("Floor", (0.25, 0.25, 0.25), 0.8)
    color_variation(m["concrete"], 3.0, (0.18, 0.18, 0.18), (0.3, 0.29, 0.28))

    # Pegboard: hardboard with a 25 mm grid of holes.
    m["pegboard"] = principled("Pegboard", (0.36, 0.25, 0.15), 0.75)
    nodes, links = m["pegboard"].node_tree.nodes, m["pegboard"].node_tree.links
    tc = nodes.new("ShaderNodeTexCoord")
    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs["Scale"].default_value = 40.0
    links.new(tc.outputs["Object"], scale.inputs[0])
    frac = nodes.new("ShaderNodeVectorMath")
    frac.operation = "FRACTION"
    links.new(scale.outputs[0], frac.inputs[0])
    # Board lies in XZ, so measure the hole distance in that plane.
    dist = nodes.new("ShaderNodeVectorMath")
    dist.operation = "DISTANCE"
    sep = nodes.new("ShaderNodeSeparateXYZ")
    comb = nodes.new("ShaderNodeCombineXYZ")
    links.new(frac.outputs[0], sep.inputs[0])
    links.new(sep.outputs["X"], comb.inputs["X"])
    links.new(sep.outputs["Z"], comb.inputs["Y"])
    links.new(comb.outputs[0], dist.inputs[0])
    dist.inputs[1].default_value = (0.5, 0.5, 0.0)
    hole = nodes.new("ShaderNodeMath")
    hole.operation = "LESS_THAN"
    hole.inputs[1].default_value = 0.16
    links.new(dist.outputs["Value"], hole.inputs[0])
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["A"].default_value = (0.36, 0.25, 0.15, 1)
    mix.inputs["B"].default_value = (0.01, 0.008, 0.006, 1)
    links.new(hole.outputs[0], mix.inputs["Factor"])
    links.new(mix.outputs["Result"], nodes["Principled BSDF"].inputs["Base Color"])
    # Dark, horizontal-ish surfaces show dust; how much is per setup.
    dust = palette.get("dust", 0.0)
    if dust:
        for key in ("env_dark", "sheet", "anodized", "motor", "body", "pcb", "frame",
                    "concrete", "wood", "lack"):
            add_dust(m[key], dust)
    return m
