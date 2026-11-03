"""Linear multilayer EXR -> camera variants, class masks and sequence labels."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import math
import os
from pathlib import Path
import shutil
import socket
import struct
import threading
import time
import uuid
import zlib

import cv2
import numpy as np
from PIL import Image

from . import isp
from .paths import GCODE_CACHE, RUNS_DIR

LOG = logging.getLogger(__name__)
FAILURE_CLASSES = {'spaghetti': 4, 'detached': 5, 'layer_shift': 6, 'nozzle_blob': 7}


def _cstring(f):
    out = bytearray()
    while True:
        c = f.read(1)
        if not c:
            raise ValueError('Truncated EXR header')
        if c == b'\0':
            return out.decode('utf8')
        out += c


def read_exr(path):
    """Read named planes from interleaved scanline EXR without extra packages.

    Supports lossless NONE/ZIPS/ZIP half/float/uint, interleaved or multipart,
    including arbitrary layer names. Blender helper explicitly selects interleaved ZIP. An installed
    OpenEXR binding is used for PIZ/DWA and other EXR encodings.
    """
    try:
        return _read_scanline_exr(path)
    except NotImplementedError as exc:
        try:
            import OpenEXR
        except ImportError:
            raise ValueError(f'{path}: {exc}; render interleaved ZIP EXR or install OpenEXR') from exc
        with OpenEXR.File(str(path), separate_channels=True) as f:
            return {name: ch.pixels.astype(np.float32) for part in f.parts
                    for name, ch in part.channels.items()}


def _read_scanline_exr(path):
    with open(path, 'rb') as f:
        magic, version = struct.unpack('<II', f.read(8))
        if magic != 20000630:
            raise ValueError(f'Not an EXR: {path}')
        if version & ~(0xff | 0x400 | 0x1000):
            raise NotImplementedError('tiled/deep EXR')
        multipart = bool(version & 0x1000)
        headers = []
        while True:
            attrs = {}
            while (name := _cstring(f)):
                typ = _cstring(f)
                size, = struct.unpack('<I', f.read(4))
                data = f.read(size)
                if len(data) != size:
                    raise ValueError('Truncated EXR attribute')
                attrs[name] = (typ, data)
            if not attrs:
                break
            if attrs.get('type', ('', b'scanlineimage'))[1] != b'scanlineimage':
                raise NotImplementedError('non-scanline EXR part')
            headers.append(attrs)
            if not multipart:
                break
        tables = []
        for attrs in headers:
            compression = attrs['compression'][1][0]
            if compression not in (0, 2, 3):
                raise NotImplementedError(f'EXR compression {compression}')
            _, ymin, _, ymax = struct.unpack('<4i', attrs['dataWindow'][1])
            count = math.ceil((ymax - ymin + 1) / (16 if compression == 3 else 1))
            tables.append(struct.unpack(f'<{count}Q', f.read(8 * count)))
        result = {}
        for part_index, (attrs, offsets) in enumerate(zip(headers, tables)):
            compression = attrs['compression'][1][0]
            if compression not in (0, 2, 3):
                raise NotImplementedError(f'EXR compression {compression}')
            xmin, ymin, xmax, ymax = struct.unpack('<4i', attrs['dataWindow'][1])
            w, h = xmax - xmin + 1, ymax - ymin + 1
            if not (0 < w <= 32768 and 0 < h <= 32768):
                raise ValueError('Invalid EXR dimensions')
            from io import BytesIO
            ch = BytesIO(attrs['channels'][1])
            channels = []
            while (name := _cstring(ch)):
                kind, _, xs, ys = struct.unpack('<iB3xii', ch.read(16))
                if xs != 1 or ys != 1:
                    raise NotImplementedError('subsampled EXR channel')
                dtype = {0: '<u4', 1: '<f2', 2: '<f4'}[kind]
                channels.append((name, np.dtype(dtype)))
            planes = {name: np.empty((h, w), np.float32) for name, _ in channels}
            lines = 16 if compression == 3 else 1
            count = math.ceil(h / lines)
            bytes_per_row = sum(dtype.itemsize * w for _, dtype in channels)
            for offset in offsets:
                f.seek(offset)
                if multipart and struct.unpack('<i', f.read(4))[0] != part_index:
                    raise ValueError('EXR chunk points to the wrong part')
                y, packed = struct.unpack('<iI', f.read(8))
                n = min(lines, ymax - y + 1)
                payload = f.read(packed)
                expected = n * bytes_per_row
                if compression and packed < expected:
                    data = np.frombuffer(zlib.decompress(payload), dtype=np.uint8)
                    # Undo predictor modulo 256, then even/odd byte deinterleave.
                    delta = data.astype(np.int64) - 128
                    delta[0] = data[0]
                    data = np.cumsum(delta).astype(np.uint8)
                    raw = np.empty_like(data)
                    split = (len(data) + 1) // 2
                    raw[::2], raw[1::2] = data[:split], data[split:]
                    payload = raw.tobytes()
                if len(payload) != expected:
                    raise ValueError(f'Truncated EXR chunk in {path}')
                pos = 0
                for row in range(n):
                    for name, dtype in channels:
                        size = w * dtype.itemsize
                        planes[name][y - ymin + row] = np.frombuffer(payload, dtype=dtype, count=w, offset=pos)
                        pos += size
            result.update(planes)
        return result


def _channel(planes, pass_name, component=None):
    candidates = []
    for name in planes:
        bits = name.split('.')
        if any(b.lower() in ({'indexob', 'object index'} if pass_name == 'IndexOB' else {pass_name.lower()}) for b in bits):
            if component is None or bits[-1].lower() == component.lower():
                candidates.append(name)
    if not candidates:
        raise ValueError(f'Missing {pass_name}.{component or "*"}; EXR channels: {sorted(planes)}')
    if len(candidates) > 1:
        raise ValueError(f'Ambiguous {pass_name} channels: {candidates}')
    return planes[candidates[0]]


def read_render(path):
    planes = read_exr(path)
    LOG.info('EXR %s channels: %s', path, ', '.join(sorted(planes)))
    rgb = np.stack([_channel(planes, 'Combined', c) for c in 'RGB'], axis=-1)
    index = _channel(planes, 'IndexOB')
    depth = _channel(planes, 'Depth')
    index = np.rint(np.nan_to_num(index, nan=1., posinf=1., neginf=1.))
    mask = np.where((index >= 0) & (index <= 8), index, 1).astype(np.uint8)
    return rgb, despeckle(mask), depth


def despeckle(mask):
    """Remove isolated IndexOB speckle (CONTRACTS §5): Cycles takes the index
    from the first sample, which sometimes follows a see-through panel's
    reflection. A pixel whose four neighbours all agree on another class
    takes that class; one-pixel-wide strands survive because their own
    neighbours along the strand disagree."""
    padded = np.pad(mask, 1, mode='edge')
    up, down = padded[:-2, 1:-1], padded[2:, 1:-1]
    left, right = padded[1:-1, :-2], padded[1:-1, 2:]
    agree = (up == down) & (up == left) & (up == right) & (up != mask)
    return np.where(agree, up, mask)


def _labels(mask, frame, spec, render_pixels=None):
    """Visible labels in one output variant; never infer pixels from state."""
    total = mask.size
    threshold = max(40 * total / (render_pixels or total), .0002 * total)
    result = {key: {} for key in ('presence', 'ignore', 'severity', 'area_frac', 'boxes', 'class_pixels')}
    state = frame.get('failure_state', {})
    failure = spec.get('failure') or {}
    own_type = failure.get('type')
    active = bool(state.get('active', False)) and frame.get('t', 0) >= failure.get('onset_s', -math.inf)
    h, w = mask.shape
    radius = max(1, round(.015 * w / 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    for class_id in range(9):
        result['class_pixels'][str(class_id)] = int(np.count_nonzero(mask == class_id))
    for name, class_id in FAILURE_CLASSES.items():
        binary = (mask == class_id).astype(np.uint8)
        count = result['class_pixels'][str(class_id)]
        present = count >= threshold
        result['presence'][name] = int(present)
        result['ignore'][name] = bool(active and own_type == name and not present)
        result['area_frac'][name] = count / total
        result['severity'][name] = float(np.clip(state.get('severity', 0), 0, 1)) if present and own_type == name else 0.
        result['boxes'][name] = []
        if not present:
            continue
        # Dilation groups nearby fragments; boxes retain bounds of original pixels.
        n, components, _, _ = cv2.connectedComponentsWithStats(cv2.dilate(binary, kernel), connectivity=8)
        candidates = []
        for component in range(1, n):
            ys, xs = np.where((components == component) & (binary != 0))
            if not len(xs):
                continue
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            area = (x1 - x0) * (y1 - y0)
            if area >= (.003 * w) ** 2:
                candidates.append((len(xs), [x0 / w, y0 / h, x1 / w, y1 / h]))
        result['boxes'][name] = [box for _, box in sorted(candidates, key=lambda item: item[0], reverse=True)[:8]]
    return result


def _blacklist_alignment(spec, meta):
    """Coordinate concurrent appenders with the blacklist file's advisory lock."""
    GCODE_CACHE.mkdir(parents=True, exist_ok=True)
    record = {'timeline': spec.get('slice', {}).get('timeline'),
              'slice_hash': spec.get('slice', {}).get('slice_hash'),
              'reason': 'render alignment_ok=false', 'alignment': meta.get('alignment'),
              'job_id': spec['job_id']}
    with (GCODE_CACHE / 'blacklist.jsonl').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(record, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream, fcntl.LOCK_UN)


def _retains_raw(spec, keep_raw):
    choice = spec.get('keep_raw', spec.get('render', {}).get('keep_raw', keep_raw))
    if isinstance(choice, bool):
        return choice
    fraction = int(hashlib.sha256(spec['job_id'].encode()).hexdigest()[:16], 16) / 2**64
    return fraction < float(choice)


def process_job(job_dir, spec=None, *, keep_raw=False, heartbeat=None):
    """Publish labels last as the completion marker. Failure retains raw data."""
    job_dir = Path(job_dir)
    spec = spec if spec is not None else json.loads((job_dir / 'spec.json').read_text())
    raw = job_dir / 'raw'
    meta = json.loads((raw / 'render_meta.json').read_text())
    times = spec['keyframes']
    frames = {int(f['idx']): f for f in meta['keyframes']}
    if len(frames) != len(times) or not spec.get('isp_variants'):
        raise ValueError('Incomplete render metadata or no ISP variants')
    for i, t in enumerate(times):
        if i not in frames or float(frames[i]['t']) != float(t) or not (raw / f'k{i:03d}.exr').is_file():
            raise ValueError(f'Incomplete raw render: keyframe {i} at {t}')
    temp = job_dir / '.tmp_post'
    if temp.exists():
        shutil.rmtree(temp)
    variants = []
    for v, variant in enumerate(spec['isp_variants']):
        seed = int(variant.get('seed', spec['seed']))
        params = isp.sample_params(variant['profile'], np.random.default_rng(seed))
        params.update(variant.get('params', {}))
        params['exposure'] = float(params.get('exposure', 0)) + float(meta.get('exposure', meta.get('camera', {}).get('exposure', 0)))
        variants.append({'var': v, 'profile': variant['profile'], 'seed': seed, 'params': params})
    labels = {'job_id': spec['job_id'], 'generator_version': spec['generator_version'], 'spec': spec,
              'keyframes': [], 'variants': variants, 'windows': spec.get('windows', []),
              **{key: meta.get(key) for key in ('render_seconds', 'blender_version', 'camera', 'alignment', 'timing')},
              'alignment_ok': meta.get('alignment_ok', True) is not False and all(f.get('alignment_ok', True) is not False for f in frames.values())}
    try:
        (temp / 'frames').mkdir(parents=True)
        (temp / 'masks').mkdir()
        for i, t in enumerate(times):
            if heartbeat:
                heartbeat()
            rgb, mask, _ = read_render(raw / f'k{i:03d}.exr')
            frame = frames[i]
            events = spec.get('events', [])
            active = [e for e in events if isp.event_active(e, t)]
            keyframe = dict(frame, idx=i, t=t, variants=[], events_active=active)
            # Preserve source metadata, including source events, across raw deletion.
            keyframe['render_events_active'] = frame.get('events_active', [])
            keyframe.setdefault('class_pixels', {str(c): int(np.count_nonzero(mask == c)) for c in range(9)})
            for key in ('alignment_ok', 'alignment'):
                keyframe.setdefault(key, meta.get(key, True if key == 'alignment_ok' else {}))
            timing = meta.get('timing', {}).get('keyframes', [])
            keyframe.setdefault('timing', timing[i] if i < len(timing) else {})
            labels['keyframes'].append(keyframe)
            for variant in variants:
                v = variant['var']
                variant['params']['native_resolution'] = variant['params'].get('native_resolution', isp.PROFILES[variant['profile']]['resolution'])
                variant['params']['resolution'] = list(rgb.shape[1::-1])
                p = dict(variant['params'], time_s=t)
                rng = np.random.default_rng(np.random.SeedSequence([variant['seed'], i]))
                image = isp.apply(rgb, p, rng, frame_index=i, events_active=events)
                warped = isp.apply_geometric(mask, p, events, t)
                variant_labels = _labels(warped, frame, spec, mask.size)
                keyframe['variants'].append(variant_labels)
                if v == 0:
                    keyframe.update({key: value for key, value in variant_labels.items() if key != 'class_pixels'})
                (temp / 'frames' / f'k{i:03d}_v{v}.jpg').write_bytes(isp.encode_jpeg(image, p['jpeg_quality'], p['subsampling']))
                Image.fromarray(warped).save(temp / 'masks' / f'k{i:03d}_v{v}.png')
        if not labels['alignment_ok']:
            _blacklist_alignment(spec, meta)
        (temp / 'labels.json').write_text(json.dumps(labels, indent=2, allow_nan=False))
        if heartbeat:
            heartbeat()
        for name in ('frames', 'masks'):
            target = job_dir / name
            if target.exists():
                shutil.rmtree(target)
            os.replace(temp / name, target)
        os.replace(temp / 'labels.json', job_dir / 'labels.json')
        temp.rmdir()
        if not (keep_raw is True or _retains_raw(spec, keep_raw)):
            shutil.rmtree(raw)
        return labels
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _run_worker(run_dir, keep_raw):
    from .manifest import Manifest
    run_dir = Path(run_dir)
    owner = f'post:{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}'
    failures = 0
    with Manifest(run_dir / 'manifest.sqlite') as manifest:
        manifest.reclaim_expired()
        while (claimed := manifest.claim('rendered', 'postprocessing', owner, 120)):
            job_id, spec = claimed
            stopped = threading.Event()
            lost = threading.Event()
            def beat():
                with Manifest(run_dir / 'manifest.sqlite') as db:
                    while not stopped.wait(30):
                        if not db.heartbeat(job_id, owner, 120):
                            lost.set()
                            return
            def check():
                if lost.is_set() or not manifest.heartbeat(job_id, owner, 120):
                    raise RuntimeError(f'Postprocessing lease lost: {job_id}')
            thread = threading.Thread(target=beat, daemon=True)
            thread.start()
            started = time.perf_counter()
            try:
                # Preserve raw until the manifest commit so a crash can retry.
                job = run_dir / 'jobs' / job_id
                process_job(job, spec, keep_raw=True, heartbeat=check)
                check()
                if not manifest.complete(job_id, owner, 'done', post_seconds=time.perf_counter() - started):
                    raise RuntimeError(f'Cannot complete stale lease: {job_id}')
                if not (keep_raw is True or _retains_raw(spec, keep_raw)):
                    shutil.rmtree(job / 'raw', ignore_errors=True)
            except Exception as exc:
                failures += 1
                LOG.exception('Postprocessing failed: %s', job_id)
                manifest.fail(job_id, owner, str(exc), 'rendered')
            finally:
                stopped.set()
                thread.join(timeout=2)
    return failures


def add_arguments(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--job-dir', type=Path)
    group.add_argument('--run', help='Run name under the configured data root, or absolute path')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--keep-raw', action='store_true')


def run(args):
    logging.basicConfig(level=logging.INFO)
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    if args.job_dir:
        try:
            process_job(args.job_dir, keep_raw=args.keep_raw)
        except Exception:
            LOG.exception('Postprocessing failed: %s', args.job_dir)
            return 1
        return 0
    run_dir = RUNS_DIR / args.run
    config_path = run_dir / 'run.json'
    keep = True if args.keep_raw else (json.loads(config_path.read_text()).get('keep_raw', False) if config_path.exists() else False)
    if not (run_dir / 'manifest.sqlite').is_file():
        raise FileNotFoundError(run_dir / 'manifest.sqlite')
    if args.workers == 1:
        return int(bool(_run_worker(str(run_dir), keep)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = [pool.submit(_run_worker, str(run_dir), keep) for _ in range(args.workers)]
        return int(any([f.result() for f in results]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    raise SystemExit(run(parser.parse_args()))
