"""Deterministic CPU camera simulation. RGB arrays are scene-linear on input.

Profile values retain the meanings in scenegen.camera; ranges are sampled once
per sequence/variant. Noise and OSD advance per frame without hidden state.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from functools import lru_cache
from io import BytesIO
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

cv2.setNumThreads(1)

PROFILES = {'chamber': {'resolution': (1280, 720),
             'jpeg_quality': 48,
             'shutter': 0.5,
             'sensor_width': 5.12,
             'lens': 2.3,
             'focus': 0.32,
             'fstop': 2.0,
             'glare': (0.85, 0.35),
             'distortion': 0.08,
             'dispersion': 0.035,
             'blur': 1.1,
             'vignette': 0.42,
             'wb': (0.93, 1.04, 1.07),
             'gain': 1.0,
             'noise': (0.0022, 0.0095),
             'chroma': 4.5,
             'banding': (0.025, 41.0),
             'hot_pixels': 4e-05,
             'saturation': 0.82,
             'contrast': 8.0,
             'sharpen': 0.22,
             'mono': False,
             'osd': []},
 'webcam': {'resolution': (1280, 720),
            'jpeg_quality': 72,
            'shutter': 0.5,
            'sensor_width': 4.8,
            'lens': 3.67,
            'focus': 0.55,
            'fstop': 2.4,
            'glare': (0.9, 0.25),
            'distortion': 0.035,
            'dispersion': 0.012,
            'blur': 0.8,
            'vignette': 0.72,
            'wb': (1.06, 1.0, 0.88),
            'gain': 1.0,
            'noise': (0.0011, 0.005),
            'chroma': 2.5,
            'banding': (0.012, 63.0),
            'hot_pixels': 0.0,
            'saturation': 1.05,
            'contrast': 10.0,
            'sharpen': 0.15,
            'mono': False,
            'osd': []},
 'pi_noir': {'resolution': (1024, 768),
             'jpeg_quality': 38,
             'shutter': 0.5,
             'sensor_width': 3.68,
             'fisheye_lens': 1.7,
             'focus': None,
             'glare': (0.8, 0.45),
             'distortion': 0.0,
             'dispersion': 0.03,
             'blur': 1.4,
             'vignette': 0.5,
             'wb': (1.22, 0.84, 1.12),
             'gain': 1.0,
             'noise': (0.003, 0.012),
             'chroma': 5.5,
             'banding': (0.0, 1.0),
             'hot_pixels': 6e-05,
             'saturation': 0.62,
             'contrast': 6.0,
             'sharpen': 0.3,
             'mono': False,
             'osd': []},
 'endoscope': {'resolution': (640, 480),
               'jpeg_quality': 34,
               'shutter': 0.5,
               'sensor_width': 2.4,
               'fisheye_lens': 1.25,
               'focus': None,
               'glare': (0.75, 0.55),
               'distortion': 0.0,
               'dispersion': 0.05,
               'blur': 1.6,
               'vignette': 0.25,
               'wb': (0.95, 1.06, 0.86),
               'gain': 1.0,
               'noise': (0.0024, 0.009),
               'chroma': 4.5,
               'banding': (0.0, 1.0),
               'hot_pixels': 0.0001,
               'saturation': 0.9,
               'contrast': 10.0,
               'sharpen': 0.4,
               'mono': False,
               'osd': []},
 'ir_night': {'resolution': (1280, 720),
              'jpeg_quality': 45,
              'shutter': 1.0,
              'sensor_width': 5.0,
              'lens': 3.2,
              'focus': None,
              'glare': (0.8, 0.5),
              'distortion': 0.06,
              'dispersion': 0.0,
              'blur': 1.1,
              'vignette': 0.35,
              'wb': (1.0, 1.0, 1.0),
              'gain': 1.0,
              'noise': (0.0022, 0.01),
              'chroma': 0.0,
              'banding': (0.0, 1.0),
              'hot_pixels': 8e-05,
              'saturation': 0.0,
              'contrast': 14.0,
              'sharpen': 0.35,
              'mono': True,
              'osd': [('2026-10-18  03:14:27', 'Left', 'Top'), ('PRINTER', 'Right', 'Bottom')]}}

# Bounds are multiplicative except for the explicitly absolute controls below.
for _p in PROFILES.values():
    _p['exposure'] = 0.0
    _p['tone_curve'] = 'srgb'
    _p['subsampling'] = 2
    _p['stream_scale'] = 1.0
    _p['timestamp_start'] = '2026-10-18T03:14:27'
    _p['ranges'] = {
        'jpeg_quality': [max(15, _p['jpeg_quality'] - 12), min(95, _p['jpeg_quality'] + 12)],
        'exposure': [-0.45, 0.45], 'gain': [0.85, 1.15],
        'blur': [0.7, 1.4], 'distortion': [0.7, 1.3], 'dispersion': [0.6, 1.4],
        'vignette': [max(0.1, _p['vignette'] - 0.12), min(1., _p['vignette'] + 0.12)],
        'wb': [0.94, 1.06], 'noise': [0.7, 1.4], 'chroma': [0.6, 1.4],
        'glare': [0.75, 1.25], 'banding': [0.7, 1.3], 'hot_pixels': [0.5, 1.5],
        'saturation': [0.85, 1.15], 'contrast': [0.8, 1.2], 'sharpen': [0.7, 1.4],
    }


def sample_params(profile, rng):
    """Resolve all sequence-level randomness into a JSON-serializable dict."""
    p = deepcopy(PROFILES[profile])
    ranges = p.pop('ranges')
    for key, bounds in ranges.items():
        factor = float(rng.uniform(*bounds))
        if key in ('jpeg_quality', 'exposure', 'vignette'):
            p[key] = int(round(factor)) if key == 'jpeg_quality' else factor
        elif isinstance(p[key], (tuple, list)):
            # Vary tuple controls (e.g. shot/read and glow threshold/energy) independently.
            p[key] = [float(v * rng.uniform(*bounds)) for v in p[key]]
        else:
            p[key] *= factor
    p['profile'] = profile
    p['subsampling'] = int(rng.choice([2, 2, 2, 2, 1, 0]))
    p['stream_scale'] = float(rng.choice([1., 1., 1., 0.75, 0.5]))
    p['band_phase'] = float(rng.uniform(0, 2 * math.pi))
    p['timestamp_start'] = (datetime(2022, 1, 1) + timedelta(seconds=int(rng.integers(0, 5 * 365 * 86400)))).isoformat()
    p['crop_offset'] = [float(x) for x in rng.uniform(-.08, .08, 2)]
    return p


def _event_params(event):
    p = dict(event.get('params', {}))
    rng = np.random.default_rng(int(p.get('seed', 0)))
    if event['type'] == 'ae_jump':
        p.setdefault('ev', float(rng.uniform(-0.8, 0.8)))
        p.setdefault('adaptation_s', 3.)
    elif event['type'] == 'wb_shift':
        p.setdefault('gains', [float(x) for x in rng.uniform(0.8, 1.2, 3)])
    elif event['type'] == 'bump':
        for k, v in zip(('dx', 'dy', 'rotation_deg', 'scale'),
                        [rng.uniform(-.025, .025), rng.uniform(-.025, .025), rng.uniform(-2, 2), rng.uniform(.98, 1.03)]):
            p.setdefault(k, float(v))
    return p


@lru_cache(maxsize=32)
def _maps(h, w, distortion, dispersion):
    y, x = np.mgrid[:h, :w].astype(np.float32)
    nx, ny = (x + .5 - w / 2) / (w / 2), (y + .5 - h / 2) / (h / 2)
    radius = (nx * nx + ny * ny) / 2
    # Fit pulls the corner back into the sensor; positive k is barrel distortion.
    k = distortion + dispersion
    scale = (1 + k * radius) / (1 + max(k, 0))
    return ((nx * scale * w / 2 + w / 2 - .5).astype(np.float32),
            (ny * scale * h / 2 + h / 2 - .5).astype(np.float32))


def event_active(event, t):
    """Bumps persist; every other event has a half-open active interval."""
    return event['t0'] <= t and (event['type'] == 'bump' or t < event.get('t1', math.inf))


def apply_geometric(img_or_mask, params, events, t):
    """Apply common lens geometry, persistent bumps, and aspect-preserving crop.

    Masks are 2-D uint8 and always use nearest-neighbor (including lens fit).
    dx/dy are fractions of frame dimensions; dx_px/dy_px may also be supplied.
    """
    a = img_or_mask
    mask = a.ndim == 2
    interp = cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR
    h, w = a.shape[:2]
    k = float(params.get('distortion', 0))
    if k or (not mask and params.get('dispersion', 0)):
        if mask:
            a = cv2.remap(a, *_maps(h, w, k, 0.), interp, borderMode=cv2.BORDER_CONSTANT)
        else:
            d = float(params.get('dispersion', 0)) * .15
            a = np.stack([cv2.remap(a[..., c], *_maps(h, w, k, d * (1 - c)), interp,
                                   borderMode=cv2.BORDER_CONSTANT) for c in range(3)], -1)
    matrix = np.eye(3)
    for e in events:
        if e['type'] != 'bump' or t < e['t0']:
            continue
        p = _event_params(e)
        warp = np.eye(3)
        warp[:2] = cv2.getRotationMatrix2D((w / 2, h / 2), p.get('rotation_deg', p.get('angle_deg', 0)), p.get('scale', 1))
        warp[0, 2] += p.get('dx_px', p.get('dx', 0) * w)
        warp[1, 2] += p.get('dy_px', p.get('dy', 0) * h)
        matrix = warp @ matrix
    if not np.array_equal(matrix, np.eye(3)):
        a = cv2.warpPerspective(a, matrix, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT)
    native_w, native_h = params.get('native_resolution', PROFILES.get(params.get('profile'), params).get('resolution', [w, h]))
    # Fit the native sensor crop, then cover the render frame using uniform
    # scaling and a second crop. The combined ROI preserves object proportions.
    ratio = (native_w / native_h) / (w / h)
    fraction = min(ratio, 1 / ratio)
    crop_w, crop_h = max(1, round(w * fraction)), max(1, round(h * fraction))
    ox, oy = params.get('crop_offset', [0, 0])
    x = int(np.clip(round((w - crop_w) * (.5 + ox)), 0, w - crop_w))
    y = int(np.clip(round((h - crop_h) * (.5 + oy)), 0, h - crop_h))
    if (crop_w, crop_h) != (w, h):
        a = cv2.resize(a[y:y + crop_h, x:x + crop_w], (w, h), interpolation=interp)
    return a


def linear_to_srgb(a):
    a = np.maximum(a, 0)
    return np.where(a <= .0031308, a * 12.92, 1.055 * np.power(a, 1 / 2.4) - .055)


@lru_cache(maxsize=16)
def _vignette(h, w, corner):
    y, x = np.mgrid[:h, :w].astype(np.float32)
    r = np.sqrt(((x + .5) / w - .5)**2 * (w / h)**2 + ((y + .5) / h - .5)**2)
    u = np.clip((r - .35) / (.5 * math.hypot(w / h, 1) - .35), 0, 1)
    return (1 + (corner - 1) * u * u * (3 - 2 * u))[..., None]


def _osd(img, params, t):
    entries = params.get('osd', [])
    if not entries:
        return img
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    h, w = img.shape[:2]
    size = max(14, h // 28)
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', size)
    except OSError:
        font = ImageFont.load_default(size=size)
    stamp = (datetime.fromisoformat(params.get('timestamp_start', '2026-10-18T03:14:27')) + timedelta(seconds=t)).strftime('%Y-%m-%d  %H:%M:%S')
    for text, horizontal, vertical in entries:
        if text[:4].isdigit() and '-' in text:
            text = stamp
        width = draw.textlength(text, font=font)
        x = size if horizontal == 'Left' else w - size - width
        y = size if vertical == 'Top' else h - size * 2
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=(246, 246, 246))
    return np.asarray(pil)


def apply(linear_rgb, params, rng, *, frame_index, events_active):
    """Simulate a frame; use a fresh RNG seeded per variant/frame for replay.

    Events can include all scheduled events: future events are ignored. `time_s`
    is the physical keyframe time, defaulting to frame_index seconds.
    """
    p = params
    t = float(p.get('time_s', frame_index))
    a = np.nan_to_num(np.asarray(linear_rgb, dtype=np.float32), nan=0., posinf=65504., neginf=0.).copy()
    h, w = a.shape[:2]
    threshold, strength = p.get('glare', (.85, 0))
    if strength:
        highlights = np.maximum(a - threshold, 0)
        # Fog glow spreads highlight energy over multiple spatial scales.
        small = cv2.resize(highlights, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
        glow = sum(cv2.GaussianBlur(small, (0, 0), max(.2, s * w / 2560)) * weight
                   for s, weight in ((3., .2), (12., .3), (36., .5)))
        glow = cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR)
        a += glow * strength * .2
    # Frame and mask share the same crop, lens fit and persistent bump.
    a = apply_geometric(a, p, events_active, t)
    blur = p.get('blur', 0) * w / PROFILES.get(p.get('profile'), p).get('resolution', [w])[0]
    if blur > 0:
        a = cv2.GaussianBlur(a, (0, 0), max(.15, blur * .5))
    a *= _vignette(h, w, float(p.get('vignette', 1)))
    ev = float(p.get('exposure', 0))
    wb = np.array(p.get('wb', [1, 1, 1]), dtype=np.float32)
    for e in events_active:
        if not event_active(e, t):
            continue
        ep = _event_params(e)
        if e['type'] == 'ae_jump':
            # A transient over/underexposure settles exponentially within seconds.
            ev += ep.get('settled_ev', 0.) + ep['ev'] * math.exp(-(t - e['t0']) / max(.001, ep['adaptation_s']))
        elif e['type'] == 'wb_shift':
            wb *= np.asarray(ep['gains'], dtype=np.float32)
    a *= wb * (float(p.get('gain', 1)) * 2**ev)
    np.maximum(a, 0, out=a)
    y = a @ np.array([.2126, .7152, .0722], np.float32)
    if p.get('mono'):
        a = np.repeat(y[..., None], 3, axis=2)
    k, read = p.get('noise', (0, 0))
    sigma = np.sqrt(k * y + read * read)[..., None]
    # Blender uses centered uniform white noise scaled by 3.4, sigma ~= 0.98.
    grain = (rng.random((h, w, 1), dtype=np.float32) - .5) * 3.4
    if p.get('chroma', 0):
        blocks = rng.random(((h + 1) // 2, (w + 1) // 2, 3), dtype=np.float32) - .5
        chroma = cv2.resize(blocks, (w, h), interpolation=cv2.INTER_NEAREST)
        grain = grain + cv2.GaussianBlur(chroma, (0, 0), 1.) * p['chroma']
    a += grain * sigma
    amp, period = p.get('banding', (0, 1))
    if amp:
        a *= (1 + amp * np.sin(np.arange(h, dtype=np.float32) * (2 * math.pi / period) + p.get('band_phase', 0) + frame_index))[..., None, None]
    hot = p.get('hot_pixels', 0)
    if hot:
        a[rng.random((h, w)) < hot] += np.array([.9, .7, .8], np.float32)
    a = np.clip(a, 0, 1)
    # The legacy compositor performs HSV saturation and contrast in linear
    # light, followed by Standard's sRGB display transform at file output.
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * p.get('saturation', 1), 0, 1)
    a = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    c = p.get('contrast', 0) / 100
    a = (a - .5) * (1 / (1 - c)) + .5
    sharpen = p.get('sharpen', 0)
    if sharpen:
        # Blender's Box Sharpen is the 3x3 [-1,-1,-1; -1,9,-1; ...] kernel.
        a += sharpen * (a * 9 - cv2.boxFilter(a, -1, (3, 3), normalize=False))
    if p.get('tone_curve', 'srgb') == 'reinhard':
        a = np.maximum(a, 0) / (1 + np.maximum(a, 0))
    img = np.rint(np.clip(linear_to_srgb(a), 0, 1) * 255).astype(np.uint8)
    size = (w, h)
    scale = p.get('stream_scale', 1.)
    if scale != 1:
        img = cv2.resize(img, (max(1, round(size[0] * scale)), max(1, round(size[1] * scale))), interpolation=cv2.INTER_AREA)
    if img.shape[1::-1] != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    return _osd(img, p, t)


def encode_jpeg(img, quality, subsampling=2):
    """Real libjpeg encoding; subsampling is resolved once per camera variant."""
    out = BytesIO()
    Image.fromarray(img).save(out, format='JPEG', quality=int(quality), subsampling=int(subsampling))
    return out.getvalue()
