"""Real Blender EXRs exercise channel decoding and complete publication."""
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from synthpipe import postprocess as post
from synthpipe.manifest import Manifest

FIXTURES = Path(__file__).parent / 'fixtures' / 'exr'
SETUPS = {'enclosed_corexy': 'chamber', 'prusa_desk': 'webcam', 'voron_topcam': 'pi_noir', 'lack_ir_night': 'ir_night'}


def make_job(path, setup='prusa_desk', seed=123):
    raw = path / 'raw'
    raw.mkdir(parents=True)
    shutil.copyfile(FIXTURES / f'{setup}.exr', raw / 'k000.exr')
    spec = {'job_id': path.name, 'seed': seed, 'generator_version': '0.1.0', 'setup': setup,
            'keyframes': [12.], 'windows': [], 'events': [],
            'isp_variants': [{'profile': SETUPS[setup], 'seed': 32, 'params': {'resolution': [64, 36]}},
                             {'profile': SETUPS[setup], 'seed': 33, 'params': {'resolution': [64, 36]}}]}
    (path / 'spec.json').write_text(json.dumps(spec))
    (raw / 'render_meta.json').write_text(json.dumps({'job_id': path.name, 'keyframes': [
        {'idx': 0, 't': 12., 'telemetry': {'layer': 3}, 'failure_state': {'active': False}}]}))
    return spec


@pytest.mark.parametrize('setup', SETUPS)
def test_real_exr_roundtrip(tmp_path, setup):
    job = tmp_path / setup
    spec = make_job(job, setup)
    channels = post.read_exr(job / 'raw' / 'k000.exr')
    assert 'ViewLayer.Combined.R' in channels
    assert 'ViewLayer.Object Index.X' in channels
    assert 'ViewLayer.Depth.Z' in channels
    rgb, mask, depth = post.read_render(job / 'raw' / 'k000.exr')
    assert rgb.dtype == np.float32 and mask.dtype == np.uint8
    assert rgb.shape[:2] == mask.shape == depth.shape
    assert np.isfinite(rgb).all() and rgb.max() > 0
    labels = post.process_job(job, keep_raw=True)
    before = (job / 'frames' / 'k000_v0.jpg').read_bytes()
    post.process_job(job, keep_raw=True)
    assert (job / 'frames' / 'k000_v0.jpg').read_bytes() == before
    assert labels['spec'] == spec
    for v in range(2):
        assert Image.open(job / 'frames' / f'k000_v{v}.jpg').size == rgb.shape[1::-1]
        assert np.asarray(Image.open(job / 'masks' / f'k000_v{v}.png')).dtype == np.uint8
    assert labels['keyframes'][0]['presence'] == dict.fromkeys(post.FAILURE_CLASSES, 0)
    post.process_job(job)
    assert not (job / 'raw').exists() and not (job / '.tmp_post').exists()


def test_incomplete_raw_preserved(tmp_path):
    spec = make_job(tmp_path / 'incomplete')
    job = tmp_path / 'incomplete'
    spec['keyframes'].append(20.)
    with pytest.raises(ValueError, match='Incomplete'):
        post.process_job(job, spec)
    assert (job / 'raw' / 'k000.exr').exists()
    assert not (job / 'labels.json').exists()


def test_presence_boxes_visibility_and_shift():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:30, 10:20] = 4
    mask[40:42, 40:42] = 5
    mask[50:60, 50:60] = 6
    frame = {'t': 12, 'failure_state': {'active': True, 'visible': True, 'type': 'detached', 'severity': .6, 'shift_layer_printed': False}}
    spec = {'failure': {'type': 'detached', 'onset_s': 10}}
    labels = post._labels(mask, frame, spec)
    assert labels['presence'] == {'spaghetti': 1, 'detached': 0, 'layer_shift': 1, 'nozzle_blob': 0}
    assert labels['ignore']['detached']
    assert labels['severity'] == dict.fromkeys(post.FAILURE_CLASSES, 0.)
    assert labels['area_frac']['spaghetti'] == .01
    assert labels['boxes']['spaghetti'] == [[.1, .2, .2, .3]]
    assert labels['boxes']['detached'] == []
    frame['t'] = 9
    assert not post._labels(mask, frame, spec)['ignore']['detached']
    frame['t'] = 12
    mask[40:50, 40:50] = 5
    assert post._labels(mask, frame, spec)['severity']['detached'] == .6


def test_threshold_scales_from_render_resolution():
    mask = np.zeros((200, 200), np.uint8)
    mask[:10, :10] = 7
    frame = {'t': 2, 'failure_state': {'active': True, 'severity': 1}}
    spec = {'failure': {'type': 'nozzle_blob', 'onset_s': 1}}
    assert post._labels(mask, frame, spec)['presence']['nozzle_blob'] == 1
    assert post._labels(mask, frame, spec, render_pixels=10000)['presence']['nozzle_blob'] == 0


def test_fragment_boxes_merge_and_cap():
    mask = np.zeros((500, 1000), np.uint8)
    mask[10:20, 10:20] = 4
    mask[10:20, 25:35] = 4
    for x in range(100, 1000, 70):
        mask[100:110, x:x + 10] = 4
    labels = post._labels(mask, {}, {})
    assert len(labels['boxes']['spaghetti']) == 8
    assert labels['boxes']['spaghetti'][0] == [.01, .02, .035, .04]
    assert np.count_nonzero(mask == 4) == labels['class_pixels']['4']


def test_variant_labels_metadata_and_blacklist(tmp_path, monkeypatch):
    job = tmp_path / 'variants'
    spec = make_job(job)
    spec['failure'] = {'type': 'detached', 'onset_s': 10}
    spec['slice'] = {'timeline': '/cache/timeline.npz', 'slice_hash': 'abc'}
    spec['events'] = [{'type': 'wb_shift', 't0': 10, 't1': 12}, {'type': 'bump', 't0': 10, 't1': 11, 'params': {'dx_px': 0, 'dy_px': 0, 'rotation_deg': 0, 'scale': 1}}]
    spec['isp_variants'][0]['params'] = {'distortion': 0, 'dispersion': 0, 'native_resolution': [2, 1]}
    spec['isp_variants'][1]['params'] = {'distortion': 0, 'dispersion': 0, 'crop_offset': [0, 0], 'native_resolution': [4, 3]}
    meta = {'keyframes': [{'idx': 0, 't': 12., 'telemetry': {'layer': 3},
            'failure_state': {'active': True, 'type': 'detached', 'severity': .7, 'extra': 'retained'},
            'class_pixels': {'5': 300}, 'events_active': ['source-event']}],
            'alignment_ok': False, 'alignment': {'ok': False, 'fit': [.2]}, 'camera': {'profile': 'webcam'},
            'blender_version': '5.2.1', 'render_seconds': 4, 'timing': {'keyframes': [{'render_s': 2}]}}
    (job / 'raw' / 'render_meta.json').write_text(json.dumps(meta))
    mask = np.zeros((100, 200), np.uint8)
    mask[20:50, :10] = 5
    monkeypatch.setattr(post, 'read_render', lambda _: (np.repeat((mask / 5)[..., None], 3, axis=2).astype(np.float32), mask, mask))
    monkeypatch.setattr(post, 'GCODE_CACHE', tmp_path / 'cache')
    labels = post.process_job(job, spec, keep_raw=True)
    k = labels['keyframes'][0]
    assert k['variants'][0]['presence']['detached'] == 1
    assert k['variants'][1]['presence']['detached'] == 0
    assert k['variants'][1]['ignore']['detached']
    assert k['presence'] == k['variants'][0]['presence']
    assert k['severity']['detached'] == .7
    assert k['class_pixels'] == {'5': 300}
    assert k['failure_state'] == meta['keyframes'][0]['failure_state']
    assert k['render_events_active'] == ['source-event']
    assert [e['type'] for e in k['events_active']] == ['bump']
    assert k['timing'] == {'render_s': 2}
    assert not labels['alignment_ok'] and not k['alignment_ok']
    assert labels['camera'] == meta['camera'] and labels['render_seconds'] == 4
    entry = json.loads((tmp_path / 'cache' / 'blacklist.jsonl').read_text())
    assert entry['slice_hash'] == 'abc' and entry['timeline'] == '/cache/timeline.npz'
    assert Image.open(job / 'frames' / 'k000_v1.jpg').size == (200, 100)


def test_manifest_success_and_failure_retries(tmp_path):
    run = tmp_path / 'run'
    good = make_job(run / 'jobs' / 'good')
    bad = make_job(run / 'jobs' / 'bad')
    (run / 'jobs' / 'bad' / 'raw' / 'k000.exr').unlink()
    with Manifest(run / 'manifest.sqlite') as db:
        db.add_jobs([good, bad])
        db.db.execute("UPDATE jobs SET state='rendered'")
    assert post._run_worker(run, False) == 3
    with Manifest(run / 'manifest.sqlite') as db:
        rows = {r['job_id']: dict(r) for r in db.db.execute('SELECT * FROM jobs')}
    assert rows['good']['state'] == 'done' and rows['good']['post_seconds'] > 0
    assert rows['bad']['state'] == 'failed' and rows['bad']['attempts'] == 3
    assert (run / 'jobs' / 'good' / 'labels.json').exists()
    assert not (run / 'jobs' / 'good' / 'raw').exists()


def test_channel_alias_and_unknown_class(monkeypatch):
    shape = (2, 3)
    planes = {f'Layer.Combined.{c}': np.ones(shape, np.float32) for c in 'RGB'}
    planes['Layer.IndexOB.X'] = np.array([[0, 4.1, 8], [9, -1, np.nan]])
    planes['Layer.Depth.Z'] = np.ones(shape, np.float32)
    monkeypatch.setattr(post, 'read_exr', lambda _: planes)
    _, mask, _ = post.read_render('unused')
    assert mask.tolist() == [[0, 4, 8], [1, 1, 1]]


def test_missing_exr_pass_fails_cleanly(tmp_path):
    job = tmp_path / 'broken'
    make_job(job)
    (job / 'raw' / 'k000.exr').write_bytes(b'not an exr')
    with pytest.raises((ValueError, __import__('struct').error)):
        post.process_job(job)
    assert not (job / '.tmp_post').exists()
    assert (job / 'raw').exists()


def test_lease_loss_does_not_publish(tmp_path):
    job = tmp_path / 'lost'
    make_job(job)
    def lost():
        raise RuntimeError('lease lost')
    with pytest.raises(RuntimeError, match='lease lost'):
        post.process_job(job, heartbeat=lost)
    assert (job / 'raw').exists()
    assert not (job / '.tmp_post').exists()
    assert not (job / 'labels.json').exists()


def test_fractional_raw_retention():
    assert not post._retains_raw({'job_id': 'one'}, 0.)
    assert post._retains_raw({'job_id': 'one'}, 1.)
    assert post._retains_raw({'job_id': 'one'}, .25) == post._retains_raw({'job_id': 'one'}, .25)


def test_blender_multipart_exr():
    rgb, mask, depth = post.read_render(FIXTURES / 'prusa_multipart.exr')
    assert rgb.shape == (36, 64, 3)
    assert mask.shape == depth.shape == (36, 64)
    assert rgb.max() > 1 and np.isfinite(rgb).all()
    assert 3 in np.unique(mask)


def test_blacklist_concurrent_appends(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(post, 'GCODE_CACHE', tmp_path)
    def append(i):
        post._blacklist_alignment({'job_id': str(i), 'slice': {'timeline': '/timeline.npz', 'slice_hash': 'slice'}}, {'alignment': {'ok': False}})
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append, range(32)))
    records = [json.loads(line) for line in (tmp_path / 'blacklist.jsonl').read_text().splitlines()]
    assert {r['job_id'] for r in records} == {str(i) for i in range(32)}
    assert len(records) == 32
