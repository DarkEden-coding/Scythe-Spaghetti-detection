import numpy as np
import pytest
import torch

from scythetrain.data import INPUT, TARGET_SUB, BucketBatchSampler, grid, gpu_augment
from scythetrain.losses import auroc, average_precision, recall_at_fpr
from scythetrain.model import TemporalFailureNet

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_sampler_buckets_and_resume():
    buckets = np.r_[np.zeros(50, int), np.ones(30, int)]
    s = BucketBatchSampler(buckets, 8, seed=3)
    s.set_epoch(2)
    full = list(s)
    assert all(len(set(buckets[b])) == 1 for b in full)
    assert len(full) == 50 // 8 + 30 // 8
    s.set_epoch(2, skip=3)
    assert list(s) == full[3:]
    s.set_epoch(3)
    assert list(s) != full


def test_weighted_sampler_respects_weights():
    buckets = np.zeros(1000, int)
    w = np.r_[np.full(10, 1.0), np.zeros(990)]
    s = BucketBatchSampler(buckets, 10, weights=w)
    assert set(np.concatenate(list(s))) <= set(range(10))


@cuda
def test_augment_keeps_frames_and_targets_aligned():
    """A bright square in every frame and a target over the same square
    must still coincide after zoom/flip."""
    torch.manual_seed(0)
    B, (sw, sh) = 6, (640, 360)
    frames = torch.zeros(B, 4, 3, sh, sw, dtype=torch.uint8, device="cuda")
    frames[..., 100:200, 400:520] = 255
    gh, gw = grid(0)
    tgt = torch.zeros(B, 4, gh * TARGET_SUB, gw * TARGET_SUB, device="cuda")
    tgt[:, 0, round(100 / sh * gh * TARGET_SUB):round(200 / sh * gh * TARGET_SUB),
        round(400 / sw * gw * TARGET_SUB):round(520 / sw * gw * TARGET_SUB)] = 1
    valid = torch.ones(B, 4, dtype=torch.bool, device="cuda")
    x, patches = gpu_augment(frames, valid, tgt, True, 0)
    W, H = INPUT[0]
    assert x.shape == (B, 4, 3, H, W) and patches.shape == (B, 4, gh, gw)
    bright = (x[:, 0, 0] > 1.0).float()
    pooled = torch.nn.functional.avg_pool2d(bright[:, None], 14)[:, 0]
    for b in range(B):
        hot = patches[b, 0] > 0.5
        assert hot.any()
        # Edge patches can be half covered after the zoom; a flip or shift
        # mismatch would leave the hot patches dark.
        assert (pooled[b][hot] > 0.3).float().mean() > 0.95
        # Every frame of a window gets the same geometry (per-frame exposure
        # jitter may flip a few antialiased edge pixels).
        assert ((x[b, 0] > 1.0) != (x[b, 3] > 1.0)).float().mean() < 0.005


@cuda
def test_model_shapes_and_history_mask():
    m = TemporalFailureNet(pretrained=False, frozen_blocks=10).cuda().eval()
    B, (W, H) = 2, INPUT[1]
    x = torch.randn(B, 4, 3, H, W, device="cuda")
    valid = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool, device="cuda")
    gaps = torch.tensor([[0, 10, 60, 300.0], [0, 0, 0, 0]], device="cuda")
    tel, has = torch.zeros(B, 4, device="cuda"), torch.ones(B, device="cuda")
    # Make fusion non-identity so masking is actually exercised.
    for p in m.fusion.parameters():
        torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad():
        logits, heat = m(x, valid, gaps, tel, has)
        assert logits.shape == (B, 4) and heat.shape == (B, 4, H // 14, W // 14)
        x2 = x.clone()
        x2[1, 1:] = torch.randn_like(x2[1, 1:])  # masked slots must not matter
        logits2, _ = m(x2, valid, gaps, tel, has)
    assert torch.allclose(logits[1], logits2[1], atol=1e-4)
    assert not torch.allclose(logits[0], logits2[0]) or torch.equal(x[0], x2[0])


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert auroc(np.array([.1, .2, .8, .9]), y) == 1.0
    assert average_precision(np.array([.1, .2, .8, .9]), y) == 1.0
    assert auroc(np.array([.9, .8, .2, .1]), y) == 0.0
    rec, thr = recall_at_fpr(np.r_[np.linspace(0, .5, 2000), .7, .4], np.r_[np.zeros(2000), 1, 1], 0.001)
    assert rec == 0.5 and .49 < thr < .5


def test_auroc_ties_count_half():
    assert auroc(np.array([1.0, 1.0]), np.array([1, 0])) == 0.5
    assert auroc(np.array([0.2, 0.9, 0.9, 0.1]), np.array([0, 1, 0, 0])) == (1 + 1 + 0.5) / 3


def test_param_groups_final_norm_full_rate():
    m = TemporalFailureNet(pretrained=False, frozen_blocks=6)
    groups = m.param_groups(1.0, encoder_lr_scale=0.1, layer_decay=0.8)
    norm_ids = {id(p) for p in m.encoder.norm.parameters()}
    rates = {g["lr"] for g in groups if any(id(p) in norm_ids for p in g["params"])}
    assert rates == {0.1 / 0.8}
