"""Run with Blender -b -P this_file -- --output /path [--setup NAME].

Builds the existing showcase scenes, renders linear ZIP multilayer EXRs and
matched compositor references from the same scene and render settings.
"""
import argparse
import json
from pathlib import Path
import sys

import bpy

BLENDER = Path(__file__).resolve().parents[2] / 'blender'
sys.path.insert(0, str(BLENDER))
import build_scene
from scenegen.camera import PROFILES

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--setup', choices=list(build_scene.SETUPS))
p.add_argument('--seed', type=int, default=3)
p.add_argument('--fixture-dir', help='Also render tiny real EXRs for regression tests')
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
for name in ([a.setup] if a.setup else ['enclosed_corexy', 'prusa_desk', 'voron_topcam', 'lack_ir_night']):
    sys.argv = ['blender', '--', '--setup', name, '--seed', str(a.seed), '--samples', '32']
    bpy.context.scene.render.image_settings.media_type = 'IMAGE'
    build_scene.main()
    scene = bpy.context.scene
    profile = next(obj for obj in bpy.data.node_groups if obj.name.startswith('CameraISP'))
    scene.render.resolution_x = 640
    camera_profile = {'enclosed_corexy':'chamber', 'prusa_desk':'webcam', 'voron_topcam':'pi_noir', 'lack_ir_night':'ir_night'}[name]
    w, h = PROFILES[camera_profile]['resolution']
    scene.render.resolution_y = round(640 * h / w)
    scene.view_layers[0].use_pass_object_index = True
    scene.view_layers[0].use_pass_z = True
    out = Path(a.output) / name
    out.mkdir(parents=True, exist_ok=True)
    # First save the raw EXR, with no compositor at all.
    scene.compositing_node_group = None
    scene.render.image_settings.media_type = 'MULTI_LAYER_IMAGE'
    scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
    scene.render.image_settings.use_exr_interleave = True
    scene.render.image_settings.color_depth = '16'
    scene.render.image_settings.exr_codec = 'ZIP'
    scene.render.filepath = str(out / 'k000.exr')
    bpy.ops.render.render(write_still=True)
    # Render again for an exact existing-compositor reference. At this small
    # resolution this also exercises all profile nodes, including the OSD.
    scene.compositing_node_group = profile
    # Match OSD pixel sizing/placement at the reduced calibration resolution.
    for node in profile.nodes:
        if node.bl_idname == 'CompositorNodeStringToImage':
            node.inputs['Size'].default_value = max(14, scene.render.resolution_y // 28)
        elif node.bl_idname == 'CompositorNodeTranslate':
            node.inputs['X'].default_value *= 640 / w
            node.inputs['Y'].default_value *= scene.render.resolution_y / h
    scene.render.image_settings.media_type = 'IMAGE'
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.quality = PROFILES[camera_profile]['jpeg_quality']
    scene.render.filepath = str(out / 'reference.jpg')
    bpy.ops.render.render(write_still=True)
    (out / 'meta.json').write_text(json.dumps({'profile': camera_profile, 'exposure': scene.view_settings.exposure, 'resolution': [640, scene.render.resolution_y]}))

    if a.fixture_dir:
        fixture = Path(a.fixture_dir)
        fixture.mkdir(parents=True, exist_ok=True)
        scene.compositing_node_group = None
        scene.render.resolution_x = 64
        scene.render.resolution_y = round(64 * h / w)
        scene.render.image_settings.media_type = 'MULTI_LAYER_IMAGE'
        scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
        scene.render.filepath = str(fixture / (name + '.exr'))
        bpy.ops.render.render(write_still=True)

        if name == 'prusa_desk':
            scene.render.image_settings.use_exr_interleave = False
            scene.render.filepath = str(fixture / 'prusa_multipart.exr')
            bpy.ops.render.render(write_still=True)
            scene.render.image_settings.use_exr_interleave = True
