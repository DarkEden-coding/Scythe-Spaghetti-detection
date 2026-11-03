import collections
import json
from pathlib import Path
import numpy as np
import pytest
from synthpipe.sampler import SETUPS, sample_specs, slice_pool
from synthpipe import paths


@pytest.fixture(autouse=True)
def isolated_corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'CORPUS_DIR', tmp_path / 'corpus')


def fixture_pool(root, total=4000):
    (root / 'model.stl').write_bytes(b'fixture mesh')
    pools = {}
    for _, profile, _ in SETUPS:
        directory = root / 'gcode_cache' / profile / 'model' / 'hash'
        directory.mkdir(parents=True)
        timeline = directory/'timeline.npz'
        np.savez(timeline, meta=np.array(json.dumps(dict(total_time_s=total))))
        (directory/'model.gcode').write_text('; test\n')
        record = dict(success=True, model=dict(id='model', stl=str(root/'model.stl'), license='CC0'),
                      settings=dict(layer_height=.12, infill_pattern='gyroid', infill_density=.25),
                      gcode=str(directory/'model.gcode'))
        (directory/'slice.json').write_text(json.dumps(record))
        pools[profile] = [{**record, '_timeline': str(timeline), '_total': total}]
    return pools


def test_determinism_rules_balance(tmp_path):
    pools = fixture_pool(tmp_path)
    specs = sample_specs(2000, 42, pools)
    assert specs == sample_specs(2000, 42, pools)
    failures = [s for s in specs if s['failure']]
    assert .35 < len(failures)/len(specs) < .45
    setup_counts = collections.Counter(s['setup'] for s in specs)
    assert all(400 < c < 600 for c in setup_counts.values())
    window_times = [w['t'] for s in specs for w in s['windows']]
    assert any(t < 60 for t in window_times)
    for s in specs:
        assert 2 <= len(s['windows']) <= 3
        assert s['keyframes'] == sorted(set(s['keyframes']))
        assert len(s['keyframes']) <= 12
        for w in s['windows']:
            assert w['history'] == [round(w['t']-d, 3) for d in (10, 60, 300) if w['t'] >= d]
            assert set([w['t'], *w['history']]) <= set(s['keyframes'])
        if s['failure']:
            onset = s['failure']['onset_s']
            assert 20 <= onset < 3995
            assert s['windows'][0]['t'] < onset < s['windows'][1]['t']
            for w in s['windows'][1:]:
                assert 9.999 <= w['t']-onset <= 1500.001
        assert len(s['isp_variants']) == 3
        assert s['isp_variants'][0]['profile'] == s['isp_variants'][1]['profile']
        if s['setup'] == 'lack_ir_night':
            assert {v['profile'] for v in s['isp_variants']} == {'ir_night'}
        else:
            assert s['isp_variants'][0]['profile'] != s['isp_variants'][2]['profile']
            assert s['isp_variants'][2]['profile'] != 'ir_night'
        assert s['slice']['layer_height'] == .12
        assert s['render']['resolution'] == [640, 480 if s['setup']=='voron_topcam' else 360]
        for event in s['events']:
            assert 0 <= event['t0'] < event['t1']
            assert any(event['t0'] <= t < event['t1'] for t in s['keyframes'])
            if event['type'] == 'part_removed':
                assert s['failure'] is None


def test_short_print_and_missing_pool(tmp_path):
    specs = sample_specs(30, 1, fixture_pool(tmp_path, total=30))
    assert all(s['failure'] is None for s in specs)
    with pytest.raises(ValueError, match='No timeline-backed'):
        sample_specs(1, 1, {})


def test_actual_slice_pool(tmp_path, monkeypatch):
    fixture_pool(tmp_path)
    import synthpipe.slicing as slicing
    monkeypatch.setattr(paths, 'GCODE_CACHE', tmp_path/'gcode_cache')
    monkeypatch.setattr(slicing, 'GCODE_CACHE', tmp_path/'gcode_cache')
    records = slice_pool('prusa_mk4')
    assert len(records) == 1
    assert records[0]['_total'] == 4000
    Path(records[0]['_timeline']).unlink()
    assert slice_pool('prusa_mk4') == []


def tv(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.abs(a/a.sum() - b/b.sum()).sum() / 2


def test_class_distributions_and_events(tmp_path):
    specs = sample_specs(5000, 57, fixture_pool(tmp_path, total=8000))
    classes = [[s for s in specs if bool(s['failure']) == flag] for flag in (False, True)]
    progress = [np.histogram([w['t']/8000 for s in group for w in s['windows']],
                             bins=np.linspace(0, 1, 21))[0] for group in classes]
    counts = [np.bincount([len(s['keyframes']) for s in group], minlength=13) for group in classes]
    assert tv(*progress) < .08
    assert tv(*counts) < .08
    for group in classes:
        assert all(w['t'] <= 7995 for s in group for w in s['windows'])
        assert .08 < sum(s['windows'][0]['t'] < 180 for s in group)/len(group) < .2
    detached = [s for s in specs if s['failure'] and s['failure']['type'] == 'detached']
    assert .25 < sum(s['failure']['onset_s'] < 400 for s in detached)/len(detached) < .5
    clean = classes[0]
    removed = sum(any(e['type']=='part_removed' for e in s['events']) for s in clean)
    assert .035 < removed/len(clean) < .065
    durations = dict(hand=(5, 60), door_open=(20, 300), lights_off=(60, 3600),
                     lights_on=(60, 3600), daylight_shift=(120, 1800))
    for s in specs:
        assert s['generator_version'] == '0.3.0'
        assert len(s['code_hash']) == 64
        for event in s['events']:
            assert any(event['t0'] <= t < event['t1'] for t in s['keyframes'])
            if event['type'] in ('ae_jump', 'wb_shift', 'bump'):
                limit = 3 if event['type'] == 'ae_jump' else 8
                assert any(0 <= t-event['t0'] <= limit for t in s['keyframes'])
            if event['type'] in durations:
                low, high = durations[event['type']]
                assert low <= event['t1']-event['t0'] <= high
            if event['type'] == 'part_removed':
                assert not s['failure']
                assert 60 <= event['t1']-event['t0'] <= 600
    for kind in ('hand', 'door_open', 'lights_on', 'lights_off', 'ae_jump', 'wb_shift', 'bump'):
        rates = [sum(any(e['type']==kind for e in s['events']) for s in group)/len(group)
                 for group in classes]
        assert abs(rates[0]-rates[1]) < .025


def test_connected_split_groups_and_holdout(tmp_path, monkeypatch):
    pools = fixture_pool(tmp_path)
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    models = []
    for i in range(24):
        stl = corpus / f'{i}.stl'
        # Models 0/1 share bytes; models 1/2 share a Thing: transitive merge.
        stl.write_bytes(str(max(0, i-1)).encode())
        models.append(dict(id=str(i), stl=str(stl), thing_id=7 if i in (1, 2) else 100+i))
    (corpus/'index.json').write_text(json.dumps(models))
    monkeypatch.setattr(paths, 'CORPUS_DIR', corpus)
    for profile, pool in pools.items():
        pools[profile] = [{**pool[0], 'model': {'id': m['id']}} for m in models]
    specs = sample_specs(5000, 42, pools)
    seen = collections.defaultdict(set)
    by_model = {}
    for spec in specs:
        seen[spec['group_id']].add(spec['split'])
        by_model[spec['model']['id']] = spec['group_id']
    assert all(len(splits) == 1 for splits in seen.values())
    assert by_model['0'] == by_model['1'] == by_model['2']
    assert set(s['split'] for s in specs) == {'train', 'val', 'test'}
    heldout = sample_specs(100, 42, pools, holdout_setup='lack_ir_night')
    assert all((s['split']=='test_setup') == (s['setup']=='lack_ir_night') for s in heldout)
    assert all('heldout_setup' not in s for s in heldout)


def test_blacklist_and_move_header(tmp_path, monkeypatch):
    pools = fixture_pool(tmp_path)
    import synthpipe.slicing as slicing
    cache = tmp_path/'gcode_cache'
    monkeypatch.setattr(paths, 'GCODE_CACHE', cache)
    monkeypatch.setattr(slicing, 'GCODE_CACHE', cache)
    timeline = Path(pools['prusa_mk4'][0]['_timeline'])
    for item in ({'timeline': str(timeline), 'reason': 'alignment'},
                 {'slice_hash': 'hash', 'reason': 'alignment'}):
        (cache/'blacklist.jsonl').write_text(json.dumps(item)+'\n')
        assert slice_pool('prusa_mk4') == []
    (cache/'blacklist.jsonl').unlink()
    # Shape is sufficient; a zero-stride source compresses cheaply and the
    # sampler must read only its header, not inflate 8M move values.
    np.savez_compressed(timeline, meta=np.array(json.dumps({'total_time_s': 4000})),
                        t0=np.broadcast_to(np.float32(0), (8_000_001,)))
    assert slice_pool('prusa_mk4') == []
