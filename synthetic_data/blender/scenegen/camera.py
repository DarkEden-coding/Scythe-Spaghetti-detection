"""Cameras: lens/sensor model, render settings and the compositor ISP.

A camera profile describes one class of cheap device end to end. The ISP
group "CameraISP" runs in device order: optics -> sensor -> ISP -> encoder.
"""

import math

import bpy

from .util import FPS, aim

# Profile keys:
#   resolution, jpeg_quality, shutter (fraction of a 1/30 s frame)
#   sensor_width (mm), lens (mm) or fisheye_lens (mm, equisolid projection)
#   focus (m) / fstop for depth of field, or None for no DOF
#   glare (threshold, strength), distortion, dispersion, blur (px)
#   vignette (gain at the corner), wb (r, g, b gains), gain
#   noise (shot k, read sigma), chroma (blotchy colour noise amount)
#   banding (amplitude, period px), hot_pixels (probability)
#   saturation, contrast, sharpen, mono, osd [(text, h_align, v_align)]
PROFILES = {
    # Built-in enclosure camera of a consumer CoreXY: 1/4" sensor, ~100 deg.
    "chamber": dict(
        resolution=(1280, 720), jpeg_quality=48, shutter=0.5,
        sensor_width=5.12, lens=2.3, focus=0.32, fstop=2.0,
        glare=(0.85, 0.35), distortion=0.08, dispersion=0.035, blur=1.1,
        vignette=0.42, wb=(0.93, 1.04, 1.07), gain=1.0,
        noise=(0.0022, 0.0095), chroma=4.5, banding=(0.025, 41.0),
        hot_pixels=0.00004, saturation=0.82, contrast=8.0, sharpen=0.22,
        mono=False, osd=[]),
    # USB desk webcam (C920 class) on a mini tripod: cleaner, warm AWB.
    "webcam": dict(
        resolution=(1280, 720), jpeg_quality=72, shutter=0.5,
        sensor_width=4.8, lens=3.67, focus=0.55, fstop=2.4,
        glare=(0.9, 0.25), distortion=0.035, dispersion=0.012, blur=0.8,
        vignette=0.72, wb=(1.06, 1.0, 0.88), gain=1.0,
        noise=(0.0011, 0.005), chroma=2.5, banding=(0.012, 63.0),
        hot_pixels=0.0, saturation=1.05, contrast=10.0, sharpen=0.15,
        mono=False, osd=[]),
    # Raspberry Pi NoIR camera with a 160 deg M12 lens: no IR-cut filter, so
    # everything takes a magenta cast; heavy compression from the streamer.
    "pi_noir": dict(
        resolution=(1024, 768), jpeg_quality=38, shutter=0.5,
        sensor_width=3.68, fisheye_lens=1.7, focus=None,
        glare=(0.8, 0.45), distortion=0.0, dispersion=0.03, blur=1.4,
        vignette=0.5, wb=(1.22, 0.84, 1.12), gain=1.0,
        noise=(0.003, 0.012), chroma=5.5, banding=(0.0, 1.0),
        hot_pixels=0.00006, saturation=0.62, contrast=6.0, sharpen=0.3,
        mono=False, osd=[]),
    # USB endoscope zip-tied to the toolhead: tiny sensor, LED ring, 4:3 VGA.
    "endoscope": dict(
        resolution=(640, 480), jpeg_quality=34, shutter=0.5,
        sensor_width=2.4, fisheye_lens=1.25, focus=None,
        glare=(0.75, 0.55), distortion=0.0, dispersion=0.05, blur=1.6,
        vignette=0.25, wb=(0.95, 1.06, 0.86), gain=1.0,
        noise=(0.0024, 0.009), chroma=4.5, banding=(0.0, 1.0),
        hot_pixels=0.0001, saturation=0.9, contrast=10.0, sharpen=0.4,
        mono=False, osd=[]),
    # Security camera in night mode: IR illuminator, monochrome, slow shutter,
    # strong noise reduction blur, timestamp overlay.
    "ir_night": dict(
        resolution=(1280, 720), jpeg_quality=45, shutter=1.0,
        sensor_width=5.0, lens=3.2, focus=None,
        glare=(0.8, 0.5), distortion=0.06, dispersion=0.0, blur=1.1,
        vignette=0.35, wb=(1.0, 1.0, 1.0), gain=1.0,
        noise=(0.0022, 0.01), chroma=0.0, banding=(0.0, 1.0),
        hot_pixels=0.00008, saturation=0.0, contrast=14.0, sharpen=0.35,
        mono=True, osd=[("2026-10-18  03:14:27", "Left", "Top"),
                        ("PRINTER", "Right", "Bottom")]),
}


def build_camera(name, profile, coll, location, target, parent=None, roll=0.0):
    p = PROFILES[profile]
    cam = bpy.data.cameras.new(name)
    cam.sensor_fit = "HORIZONTAL"
    cam.sensor_width = p["sensor_width"]
    cam.clip_start = 0.002
    if "fisheye_lens" in p:
        cam.type = "PANO"
        cam.panorama_type = "FISHEYE_EQUISOLID"
        cam.fisheye_lens = p["fisheye_lens"]
        cam.fisheye_fov = math.pi
    else:
        cam.lens = p["lens"]
    if p.get("focus"):
        cam.dof.use_dof = True
        cam.dof.focus_distance = p["focus"]
        cam.dof.aperture_fstop = p["fstop"]
    obj = bpy.data.objects.new(name, cam)
    obj.location = location
    coll.objects.link(obj)
    obj.parent = parent
    aim(obj, target, roll)
    bpy.context.scene.camera = obj
    return obj


def setup_render(profile, samples, exposure):
    p = PROFILES[profile]
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.cycles.blur_glossy = 1.0
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
        except TypeError:
            continue
        if any(d.type == backend for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == backend
            scene.cycles.device = "GPU"
            break

    scene.render.resolution_x, scene.render.resolution_y = p["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.frame_start = scene.frame_end = scene.frame_current = 1
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = p["shutter"]
    scene.render.filepath = "//../renders/frame_"

    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = exposure
    img = scene.render.image_settings
    img.media_type = "IMAGE"
    img.file_format = "JPEG"
    img.color_mode = "RGB"
    img.quality = p["jpeg_quality"]


def setup_isp(profile, seed):
    """Compositor group that turns a clean render into a cheap camera frame."""
    p = PROFILES[profile]
    scene = bpy.context.scene
    tree = bpy.data.node_groups.new("CameraISP", "CompositorNodeTree")
    tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    scene.compositing_node_group = tree
    N, L = tree.nodes, tree.links
    x = [0]

    def node(kind, **inputs):
        n = N.new(kind)
        n.location = (x[0], 0)
        x[0] += 220
        for key, value in inputs.items():
            n.inputs[key].default_value = value
        return n

    def math_node(op, *args, vector=False):
        n = node("ShaderNodeVectorMath" if vector else "ShaderNodeMath")
        n.operation = op
        if vector:
            slots = [n.inputs[0], n.inputs["Scale"]] if op == "SCALE" else list(n.inputs)
            scalar_out = op in ("DOT_PRODUCT", "LENGTH", "DISTANCE")
            out = n.outputs["Value"] if scalar_out else n.outputs["Vector"]
        else:
            slots, out = list(n.inputs), n.outputs[0]
        for sock, v in zip(slots, args):
            if isinstance(v, bpy.types.NodeSocket):
                L.new(v, sock)
            else:
                sock.default_value = v
        return out

    rl = node("CompositorNodeRLayers")
    res_x, res_y = p["resolution"]

    # --- optics -----------------------------------------------------------
    glare = node("CompositorNodeGlare", Type="Fog Glow", Quality="High",
                 Threshold=p["glare"][0], Strength=p["glare"][1], Size=0.7)
    L.new(rl.outputs["Image"], glare.inputs["Image"])
    img = glare.outputs["Image"]
    if p["distortion"] or p["dispersion"]:
        lens = node("CompositorNodeLensdist", Type="Radial", Distortion=p["distortion"],
                    Dispersion=p["dispersion"], Jitter=False, Fit=True)
        L.new(img, lens.inputs["Image"])
        img = lens.outputs["Image"]
    soft = node("CompositorNodeBlur", Type="Gaussian", Size=(p["blur"], p["blur"]))
    L.new(img, soft.inputs["Image"])
    optics = soft.outputs["Image"]

    coords = node("CompositorNodeImageCoordinates")
    L.new(optics, coords.inputs["Image"])
    # Vignette: smooth falloff on aspect-corrected radius.
    centred = math_node("SUBTRACT", coords.outputs["Normalized"], (0.5, 0.5, 0.0), vector=True)
    aspect = math_node("MULTIPLY", centred, (res_x / res_y, 1.0, 0.0), vector=True)
    radius = math_node("LENGTH", aspect, vector=True)
    vig = node("ShaderNodeMapRange")
    L.new(radius, vig.inputs["Value"])
    vig.inputs["From Min"].default_value = 0.35
    vig.inputs["From Max"].default_value = 0.5 * math.hypot(res_x / res_y, 1.0)
    vig.inputs["To Min"].default_value = 1.0
    vig.inputs["To Max"].default_value = p["vignette"]
    vig.interpolation_type = "SMOOTHSTEP"
    img = math_node("SCALE", optics, vig.outputs["Result"], vector=True)

    # --- sensor -----------------------------------------------------------
    gains = tuple(g * p["gain"] for g in p["wb"])
    img = math_node("MULTIPLY", img, gains, vector=True)
    img = math_node("MAXIMUM", img, (0.0, 0.0, 0.0), vector=True)
    luma = math_node("DOT_PRODUCT", img, (0.2126, 0.7152, 0.0722), vector=True)
    if p["mono"]:
        img = math_node("SCALE", (1.0, 1.0, 1.0), luma, vector=True)

    frame = node("CompositorNodeSceneTime").outputs["Frame"]
    seed_w = math_node("ADD", frame, float(seed % 997) * 1.37)
    pixel = coords.outputs["Pixel"]

    def white_noise(vec, w_offset):
        wn = node("ShaderNodeTexWhiteNoise")
        wn.noise_dimensions = "4D"
        L.new(vec, wn.inputs["Vector"])
        L.new(math_node("ADD", seed_w, w_offset), wn.inputs["W"])
        return wn

    # Shot + read noise: sigma = sqrt(k * Y + r^2).
    k, read = p["noise"]
    sigma = math_node("SQRT", math_node("MULTIPLY_ADD", luma, k, read * read))
    lum = math_node("MULTIPLY",
                    math_node("SUBTRACT", white_noise(pixel, 0.0).outputs["Value"], 0.5), 3.4)
    grain = math_node("SCALE", (1.0, 1.0, 1.0), lum, vector=True)
    if p["chroma"]:
        # Chroma noise on 2x2 blocks, blurred into blotches (typical after ISP NR).
        block = math_node("FLOOR", math_node("MULTIPLY", pixel, (0.5, 0.5, 1.0), vector=True),
                          vector=True)
        chroma = math_node("SUBTRACT", white_noise(block, 17.0).outputs["Color"],
                           (0.5, 0.5, 0.5), vector=True)
        blotch = node("CompositorNodeBlur", Type="Gaussian", Size=(2.0, 2.0))
        L.new(chroma, blotch.inputs["Image"])
        grain = math_node("ADD", math_node("SCALE", blotch.outputs["Image"], p["chroma"],
                                           vector=True), grain, vector=True)
    img = math_node("ADD", img, math_node("SCALE", grain, sigma, vector=True), vector=True)

    # Rolling-shutter banding from PWM-dimmed LEDs.
    amp, period = p["banding"]
    if amp:
        py = math_node("DOT_PRODUCT", pixel, (0.0, 1.0, 0.0), vector=True)
        phase = math_node("MULTIPLY_ADD", py, 2 * math.pi / period, seed_w)
        band = math_node("MULTIPLY_ADD", math_node("SINE", phase), amp, 1.0)
        img = math_node("SCALE", img, band, vector=True)

    if p["hot_pixels"]:
        hot_mask = math_node("GREATER_THAN", white_noise(pixel, 53.0).outputs["Value"],
                             1.0 - p["hot_pixels"])
        img = math_node("ADD", img, math_node("SCALE", (0.9, 0.7, 0.8), hot_mask, vector=True),
                        vector=True)

    img = math_node("MINIMUM", math_node("MAXIMUM", img, (0, 0, 0), vector=True),
                    (1, 1, 1), vector=True)

    # --- ISP --------------------------------------------------------------
    sat = node("CompositorNodeHueSat", Saturation=p["saturation"], Value=1.0, Factor=1.0)
    L.new(img, sat.inputs["Image"])
    contrast = node("CompositorNodeBrightContrast", Brightness=0.0, Contrast=p["contrast"])
    L.new(sat.outputs["Image"], contrast.inputs["Image"])
    sharpen = node("CompositorNodeFilter", Type="Box Sharpen", Factor=p["sharpen"])
    L.new(contrast.outputs["Image"], sharpen.inputs["Image"])
    out_img = sharpen.outputs["Image"]

    # On-screen display, burned in after the ISP like a real camera's OSD:
    # white text with a 2 px drop shadow. StringToImage renders full-frame with
    # the text anchored at the frame centre, so shift it out to the corner.
    size = max(14, res_y // 28)
    for text, h_align, v_align in p["osd"]:
        label = node("CompositorNodeStringToImage", String=text, Size=float(size))
        label.inputs["Horizontal Alignment"].default_value = h_align
        label.inputs["Vertical Alignment"].default_value = v_align
        sx = -1 if h_align == "Left" else 1
        sy = 1 if v_align == "Top" else -1
        off_x = sx * (res_x / 2 - size)
        off_y = sy * (res_y / 2 - size * 0.8)
        for shade, dx, dy in (((0, 0, 0, 1), 2, -2), ((0.92, 0.92, 0.92, 1), 0, 0)):
            moved = node("CompositorNodeTranslate", X=off_x + dx, Y=off_y + dy)
            L.new(label.outputs["Image"], moved.inputs["Image"])
            mask = node("CompositorNodeRGBToBW")
            L.new(moved.outputs["Image"], mask.inputs["Image"])
            over = node("CompositorNodeAlphaOver", Foreground=shade)
            L.new(out_img, over.inputs["Background"])
            L.new(mask.outputs["Val"], over.inputs["Factor"])
            out_img = over.outputs["Image"]

    out = node("NodeGroupOutput")
    L.new(out_img, out.inputs[0])
