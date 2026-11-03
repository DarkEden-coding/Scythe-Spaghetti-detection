"""Deterministic scenario specs from successful, timeline-backed slices."""
import collections
import hashlib
import json
import os
import random
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from synthpipe import GENERATOR_VERSION, paths
from synthpipe.manifest import Manifest
from synthpipe.version import code_hash

SETUPS = [('enclosed_corexy', 'bambu_x1c', 'chamber'),
          ('prusa_desk', 'prusa_mk4', 'webcam'),
          ('voron_topcam', 'voron24_300', 'pi_noir'),
          ('lack_ir_night', 'bambu_a1mini', 'ir_night')]
FILAMENTS = dict(zip(
    ['orange PLA', 'galaxy grey PLA', 'white PLA', 'green PLA', 'red PETG', 'blue PLA',
     'yellow PLA', 'purple PLA', 'black PLA', 'cyan PETG', 'pink PLA', 'brown PLA',
     'silver PLA', 'lime PLA', 'ivory PLA', 'teal PETG'],
    [(0.8,.2,.03), (.1,.1,.11), (.78,.78,.74), (.02,.45,.2), (.6,.02,.03),
     (.03,.12,.55), (.85,.6,.02), (.25,.04,.45), (.015,.015,.015), (.02,.55,.7),
     (.8,.08,.3), (.22,.09,.025), (.45,.45,.48), (.3,.7,.02), (.85,.8,.6), (.02,.3,.28)]))


def run_dir(name):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Run name must be a single directory name')
    return paths.RUNS_DIR / name


def slice_pool(profile):
    try:
        from synthpipe.slicing import pool
    except ImportError:
        entries = []
        for p in sorted((paths.GCODE_CACHE / profile).glob('*/*/slice.json')):
            item = json.loads(p.read_text())
            if item.get('success') is False or item.get('status', 'ok') not in ('ok', 'success', 'done', 'sliced'):
                continue
            item.setdefault('gcode', str(p.parent / 'model.gcode'))
            item.setdefault('timeline', str(p.parent / 'timeline.npz'))
            item.setdefault('model_id', p.parent.parent.name)
            entries.append(item)
    else:
        entries = list(pool(profile))
    blacklist = paths.GCODE_CACHE / 'blacklist.jsonl'
    blocked_paths, blocked_hashes = set(), set()
    if blacklist.exists():
        for line in blacklist.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get('timeline'):
                blocked_paths.add(str(Path(item['timeline']).resolve()))
            if item.get('slice_hash'):
                blocked_hashes.add(item['slice_hash'])
    result = []
    for entry in entries:
        entry = dict(entry)
        sl = entry.get('slice', entry)
        timeline = sl.get('timeline', entry.get('timeline'))
        gcode = sl.get('gcode', entry.get('gcode'))
        if not timeline and gcode:
            timeline = str(Path(gcode).with_name('timeline.npz'))
        if not timeline or not Path(timeline).is_file():
            continue
        if (str(Path(timeline).resolve()) in blocked_paths
                or entry.get('slice_hash', Path(timeline).parent.name) in blocked_hashes):
            continue
        with np.load(timeline, allow_pickle=False) as data:
            meta = json.loads(str(data['meta'].item()))
            total = float(meta['total_time_s'])
        # NPZ mmap does not avoid decompression. Read only the NPY header from
        # the ZIP member instead of materializing millions of moves.
        with zipfile.ZipFile(timeline) as archive:
            if 't0.npy' in archive.namelist():
                with archive.open('t0.npy') as stream:
                    version = np.lib.format.read_magic(stream)
                    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                              else np.lib.format.read_array_header_2_0)
                    shape, _, _ = reader(stream)
                    moves = shape[0]
            else:
                moves = meta.get('n_moves', sum(o.get('n_moves', 0) for o in meta.get('objects', [])))
        if moves > 8_000_000 or not np.isfinite(total) or total <= 0:
            continue
        entry['_timeline'], entry['_total'] = str(Path(timeline).resolve()), total
        result.append(entry)
    return sorted(result, key=lambda e: e['_timeline'])


def _model(entry, index):
    model = dict(entry.get('model', {}))
    model.setdefault('id', entry.get('model_id', Path(entry['_timeline']).parent.parent.name))
    source = index.get(model['id'], {})
    model = {**source, **model}
    if source.get('thing_id') is not None:
        model['thing_id'] = source['thing_id']
    model.setdefault('stl', str((paths.CORPUS_DIR / 'stl' / (model['id']+'.stl')).resolve()))
    model.setdefault('license', entry.get('model_license', entry.get('license', 'unknown')))
    return model


def _groups(pools):
    """Connected components join both source Things and byte-identical STLs."""
    index_path = paths.CORPUS_DIR / 'index.json'
    index = {m['id']: m for m in json.loads(index_path.read_text())} if index_path.exists() else {}
    models = list(index.values()) + [_model(e, index) for pool in pools.values() for e in pool]
    parent, tokens = {}, {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    hashes = {}
    for model in models:
        stl = Path(model['stl'])
        if stl not in hashes:
            if not stl.is_file():
                raise ValueError(f'Missing STL for split grouping: {stl}')
            hashes[stl] = hashlib.sha256(stl.read_bytes()).hexdigest()
        keys = ['stl:' + hashes[stl]]
        if model.get('thing_id') is not None:
            keys.append('thing:' + str(model['thing_id']))
        for key in keys:
            find(key)
        roots = [find(key) for key in keys]
        for key in roots:
            parent[key] = min(roots)
        tokens[model['id']] = keys[0]
    return index, {mid: find(token) for mid, token in tokens.items()}


def _schedule(rng, total):
    """The same latent onset and windows are used for BOTH labels.

    Reject the entire candidate, never clip a post window onto print end.
    Prints too short for onset >=20 and delta >=10 use a shared clean-only
    schedule. Longer prints always retain a pre-onset and 1–2 post windows.
    """
    end = max(0, total - 5.001)
    if end <= 31:
        return None, [rng.uniform(0, end)]
    count = 1 if rng.random() < .7 else 2
    for _ in range(10000):
        anchor = rng.uniform(0, min(180, end)) if rng.random() < .1 else rng.uniform(0, end)
        onset = max(20, anchor + rng.uniform(1, 30))
        deltas = [rng.uniform(10, 90) if rng.random() < .4
                  else 90 * (1500 / 90) ** rng.random() for _ in range(count)]
        if onset + max(deltas) <= end:
            return onset, [anchor, *(onset + d for d in deltas)]
    raise ValueError(f'Cannot sample windows inside print of {total}s')


def _events(rng, setup, keyframes, total, clean):
    kinds = ['lights_off', 'lights_on', 'daylight_shift', 'hand', 'ae_jump', 'wb_shift', 'bump']
    if setup in ('enclosed_corexy', 'voron_topcam', 'lack_ir_night'):
        kinds.append('door_open')
    chosen = [rng.choice(kinds) for _ in range(rng.choice([1, 2]))] if rng.random() < .25 else []
    if clean and rng.random() < .05:
        chosen.append('part_removed')
    durations = dict(hand=(5, 60), door_open=(20, 300), lights_off=(60, 3600),
                     lights_on=(60, 3600), daylight_shift=(120, 1800),
                     part_removed=(60, 600), ae_jump=(3, 8), wb_shift=(10, 120))
    events = []
    for kind in chosen:
        frame = rng.choice(keyframes)
        t0 = max(0, frame - rng.uniform(0, 3 if kind == 'ae_jump' else
                                      8 if kind in ('wb_shift', 'bump') else 4))
        t1 = total if kind == 'bump' else t0 + rng.uniform(*durations[kind])
        # Events may outlast the sampled print, but every interval contains
        # its selected frame under the common half-open activity contract.
        events.append(dict(type=kind, t0=t0, t1=t1, params={'seed': rng.randrange(2**31)}))
    return events


def sample_specs(count, seed, pools=None, holdout_setup=None):
    if count < 0:
        raise ValueError('count must be nonnegative')
    pools = pools if pools is not None else {p: slice_pool(p) for _, p, _ in SETUPS}
    missing = [p for _, p, _ in SETUPS if not pools.get(p)]
    if missing:
        raise ValueError(f'No timeline-backed slices for: {", ".join(missing)}')
    if holdout_setup is not None and holdout_setup not in {s for s, _, _ in SETUPS}:
        raise ValueError(f'Unknown held-out setup: {holdout_setup}')
    index, groups = _groups(pools)
    fingerprint = code_hash()
    rng = random.Random(seed)
    specs = []
    for i in range(count):
        setup, profile, camera = rng.choice(SETUPS)
        entry = rng.choice(pools[profile])
        total = entry['_total']
        onset, times = _schedule(rng, total)
        failure = None
        if rng.random() < .4 and onset is not None:
            # First-layer detachment is common: raise its conditional share
            # early without changing the common window-time distribution.
            detached = .85 if onset / total < .05 else .26
            other = 1 - detached
            kind = rng.choices(['spaghetti', 'detached', 'layer_shift', 'nozzle_blob'],
                               [other * .35/.65, detached, other * .15/.65, other * .15/.65])[0]
            params = {'seed': rng.randrange(2**31)}
            if kind == 'detached':
                params['mode'] = rng.choice(['slide', 'tip', 'knockoff', 'delaminate'])
            failure = dict(type=kind, onset_s=onset, params=params)
        windows = []
        for t in sorted(set(round(t, 3) for t in times)):
            windows.append({'t': t, 'history': [round(t-d, 3) for d in (10, 60, 300) if t >= d]})
        keyframes = sorted({t for w in windows for t in [w['t'], *w['history']]})
        events = _events(rng, setup, keyframes, total, failure is None)
        sl = {**dict(layer_height=.2, infill_pattern='grid', infill_density=.15,
                      supports=False, brim=False, copies=1, scale=1.),
              **entry.get('settings', {}), **entry.get('slice', {})}
        for key in ('layer_height', 'infill_pattern', 'infill_density', 'supports', 'brim', 'copies', 'scale'):
            if key in entry:
                sl[key] = entry[key]
        sl.update(gcode=str(Path(sl.get('gcode', entry.get('gcode',
                       Path(entry['_timeline']).with_name('model.gcode')))).resolve()),
                  timeline=entry['_timeline'])
        model = _model(entry, index)
        group = groups[model['id']]
        bucket = int(hashlib.sha256(group.encode()).hexdigest(), 16) % 100
        split = 'train' if bucket < 80 else 'val' if bucket < 90 else 'test'
        if setup == holdout_setup:
            split = 'test_setup'
        name = rng.choice(list(FILAMENTS))
        # Cross-camera variant: only between cameras that could plausibly see
        # this scene. The IR scene is lit by IR alone, so it stays monochrome.
        colour = [c for _, _, c in SETUPS if c != 'ir_night']
        other = 'ir_night' if camera == 'ir_night' else rng.choice([c for c in colour if c != camera])
        specs.append(dict(job_id=f's{i:06d}', seed=rng.randrange(2**31),
            generator_version=GENERATOR_VERSION, code_hash=fingerprint, group_id=group, split=split,
            setup=setup, printer_profile=profile,
            model=model, slice=sl, filament={'name': name, 'rgb': list(FILAMENTS[name])},
            scene=dict(variation_seed=rng.randrange(2**31), camera_jitter=dict(pos_m=.015, rot_deg=2.),
                       light_scale=rng.uniform(.7, 1.3), clutter=True),
            failure=failure, keyframes=keyframes, windows=windows, events=events,
            isp_variants=[dict(profile=c, seed=rng.randrange(2**31)) for c in (camera, camera, other)],
            render=dict(samples=48, resolution=[640, 360 if camera in ('webcam', 'chamber', 'ir_night') else 480],
                        motion_blur_shutter=1. if camera == 'ir_night' else .5)))
    return specs


def sample_stats(specs):
    progress, keyframes = {}, {}
    for label in ('clean', 'failure'):
        selected = [s for s in specs if bool(s['failure']) == (label == 'failure')]
        totals = {}
        values = []
        for spec in selected:
            timeline = spec['slice']['timeline']
            if timeline not in totals:
                with np.load(timeline, allow_pickle=False) as data:
                    totals[timeline] = json.loads(str(data['meta'].item()))['total_time_s']
            values.extend(w['t'] / totals[timeline] for w in spec['windows'])
        progress[label] = np.histogram(values, bins=np.linspace(0, 1, 11))[0].tolist()
        keyframes[label] = dict(collections.Counter(len(s['keyframes']) for s in selected))
    events = {}
    for spec in specs:
        for event in spec['events']:
            item = events.setdefault(event['type'], dict(events=0, active_keyframes=0))
            item['events'] += 1
            item['active_keyframes'] += sum(event['t0'] <= t < event['t1'] for t in spec['keyframes'])
    return dict(jobs=len(specs), failures=dict(collections.Counter(
        s['failure']['type'] if s['failure'] else 'clean' for s in specs)),
        keyframes=dict(collections.Counter(len(s['keyframes']) for s in specs)),
        setups=dict(collections.Counter(s['setup'] for s in specs)),
        progress_bin_edges=np.linspace(0, 1, 11).tolist(), progress_by_class=progress,
        keyframes_by_class=keyframes, splits=dict(collections.Counter(s['split'] for s in specs)),
        events=events)


def add_arguments(parser):
    parser.add_argument('--run', required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--keep-raw', type=float, default=.02)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--holdout-setup', choices=[s for s, _, _ in SETUPS], default=None)


def run(args):
    specs = sample_specs(args.count, args.seed, holdout_setup=getattr(args, 'holdout_setup', None))
    print(json.dumps(sample_stats(specs), indent=2))
    if args.dry_run:
        return 0
    if not 0 <= args.keep_raw <= 1:
        raise ValueError('keep_raw must be in [0,1]')
    directory = run_dir(args.run)
    directory.mkdir(parents=True, exist_ok=True)
    def git(*command):
        return subprocess.check_output(['git', '-C', str(paths.REPO_ROOT), *command], text=True).strip()
    config = dict(seed=args.seed, count=args.count, keep_raw=args.keep_raw,
                  generator_version=GENERATOR_VERSION, code_hash=code_hash(),
                  holdout_setup=getattr(args, 'holdout_setup', None), git_hash=git('rev-parse', 'HEAD'),
                  git_dirty=bool(git('status', '--porcelain')),
                  render=dict(samples=48, resolution=[640, 360], motion_blur_shutter=.5,
                              resolution_by_profile={c: [640, 480 if c == 'pi_noir' else 360]
                                                     for _, _, c in SETUPS},
                              shutter_by_profile={c: 1. if c == 'ir_night' else .5
                                                  for _, _, c in SETUPS}))
    config_path = directory / 'run.json'
    # Refuse silent replacement of a run with a different deterministic job population.
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if any(previous.get(k) != config[k] for k in ('seed', 'count', 'keep_raw', 'generator_version', 'code_hash', 'holdout_setup')):
            raise ValueError('Run already exists with different sampling configuration')
    else:
        temporary_config = directory / '.run.json.tmp'
        with temporary_config.open('w') as stream:
            json.dump(config, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_config.replace(config_path)
    with Manifest(directory / 'manifest.sqlite') as manifest:
        for spec in specs:
            row = manifest.db.execute('SELECT spec FROM jobs WHERE job_id=?', (spec['job_id'],)).fetchone()
            if row and json.loads(row['spec']) != spec:
                raise ValueError('Slice pool changed; use a new run name')
        manifest.add_jobs(specs)
    temp = directory / '.specs.jsonl.tmp'
    with temp.open('w') as stream:
        stream.write(''.join(json.dumps(s, sort_keys=True)+'\n' for s in specs))
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(directory / 'specs.jsonl')
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return 0
