"""Camera reproducibility, linear-light handling and geometry regression tests."""
import json

import numpy as np
import pytest
from PIL import Image
from io import BytesIO

from synthpipe import isp


@pytest.mark.parametrize('profile', sorted(isp.PROFILES))
def test_profiles_resolve_and_replay(profile):
    p = isp.sample_params(profile, np.random.default_rng(12))
    assert json.loads(json.dumps(p)) == json.loads(json.dumps(isp.sample_params(profile, np.random.default_rng(12))))
    p['resolution'] = [96, 54]
    a = np.full((54, 96, 3), .18, np.float32)
    first = isp.apply(a, p, np.random.default_rng(99), frame_index=3, events_active=[])
    second = isp.apply(a, p, np.random.default_rng(99), frame_index=3, events_active=[])
    assert np.array_equal(first, second)
    assert first.dtype == np.uint8 and first.shape == (54, 96, 3)
    jpeg = isp.encode_jpeg(first, p['jpeg_quality'], p['subsampling'])
    assert jpeg.startswith(b'\xff\xd8')
    assert Image.open(BytesIO(jpeg)).size == (96, 54)
    if p['mono']:
        assert np.array_equal(first[..., 0], first[..., 1])
        assert np.array_equal(first[..., 1], first[..., 2])


def clean_params():
    return dict(resolution=[80, 60], glare=[1, 0], distortion=0, dispersion=0,
                blur=0, vignette=1, wb=[1, 1, 1], gain=1, noise=[0, 0], chroma=0,
                banding=[0, 1], hot_pixels=0, saturation=1, contrast=0, sharpen=0)


def test_srgb_and_exposure():
    p = clean_params()
    a = np.full((60, 80, 3), .18, np.float32)
    out = isp.apply(a, p, np.random.default_rng(0), frame_index=0, events_active=[])
    assert abs(float(out.mean()) - 118) < 1
    p['exposure'] = 1
    out = isp.apply(a, p, np.random.default_rng(0), frame_index=0, events_active=[])
    assert abs(float(out.mean()) - 162) < 1


def test_bump_mask_alignment_and_persistence():
    p = clean_params()
    mask = np.zeros((60, 80), np.uint8)
    mask[10:20, 20:30] = 7
    rgb = np.repeat((mask / 7)[..., None], 3, axis=2).astype(np.float32)
    events = [{'type': 'bump', 't0': 5, 't1': 6, 'params': {'dx_px': 4, 'dy_px': 3, 'rotation_deg': 0, 'scale': 1}}]
    assert np.array_equal(isp.apply_geometric(mask, p, events, 4), mask)
    changed = isp.apply_geometric(mask, p, events, 100)
    assert changed[13:23, 24:34].min() == 7
    warped = isp.apply_geometric(rgb, p, events, 100)
    assert np.array_equal(changed == 7, warped[..., 0] == 1)
    assert set(np.unique(changed)) == {0, 7}


def test_seed_only_events_and_ae_adaptation():
    p = clean_params()
    image = np.full((60, 80, 3), .18, np.float32)
    event = {'type': 'ae_jump', 't0': 10, 't1': 20, 'params': {'ev': 1., 'adaptation_s': 2.}}
    def frame(t):
        return isp.apply(image, dict(p, time_s=t), np.random.default_rng(0), frame_index=0, events_active=[event])
    assert frame(10).mean() > frame(12).mean() > frame(30).mean()
    assert np.array_equal(frame(0), frame(9))
    bump = [{'type': 'bump', 't0': 0, 't1': 1, 'params': {'seed': 11}}]
    assert np.array_equal(isp.apply_geometric(image, p, bump, 10), isp.apply_geometric(image, p, bump, 10))


def test_timestamp_advances():
    p = dict(clean_params(), osd=[['2026-10-18 03:14:27', 'Left', 'Top']], resolution=[640, 360])
    a = np.zeros((360, 640, 3), np.uint8)
    assert not np.array_equal(isp._osd(a, p, 0), isp._osd(a, p, 60))


@pytest.mark.parametrize('kind,params', [('ae_jump', {'ev': 1, 'settled_ev': .4}), ('wb_shift', {'gains': [1.5, .8, .7]})])
def test_post_events_half_open(kind, params):
    image = np.full((60, 80, 3), .18, np.float32)
    event = {'type': kind, 't0': 10, 't1': 20, 'params': params}
    def frame(t):
        return isp.apply(image, dict(clean_params(), time_s=t), np.random.default_rng(0), frame_index=0, events_active=[event])
    assert not np.array_equal(frame(10), frame(9))
    assert np.array_equal(frame(20), frame(9))
    assert not isp.event_active(event, 20)


@pytest.mark.parametrize('native', [(4, 3), (16, 9)])
def test_crop_preserves_shape_and_mask_alignment(native):
    mask = np.zeros((180, 320), np.uint8)
    mask[60:120, 130:190] = 7
    rgb = np.repeat((mask / 7)[..., None], 3, axis=2).astype(np.float32)
    p = dict(clean_params(), resolution=[1280, 720], native_resolution=native, crop_offset=[.07, -.04])
    warped = isp.apply_geometric(mask, p, [], 0)
    image = isp.apply(rgb, p, np.random.default_rng(0), frame_index=0, events_active=[])
    assert image.shape[:2] == warped.shape == mask.shape
    ys, xs = np.where(warped == 7)
    assert abs(np.ptp(ys) - np.ptp(xs)) <= 1
    assert np.mean((image[..., 0] > 128) == (warped == 7)) > .995
    assert set(np.unique(warped)) == {0, 7}
