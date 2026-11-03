"""Loading: CPU workers read JPEG bytes and build mask targets; the GPU
decodes (nvJPEG), resizes and augments whole batches.

Workers stay cheap on purpose: the render run shares this machine's CPU,
and JPEG decode was the measured bottleneck when done on the CPU.
"""

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from torchvision.io import decode_jpeg

from .index import BUCKETS, TYPE_CLASS

PATCH = 14
# Model input per bucket (multiples of the 14 px patch).
INPUT = ((448, 252), (448, 336))
# Targets are kept at TARGET_SUB x the patch grid so geometric augmentation
# can resample them before pooling to patches.
TARGET_SUB = 4
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1) * 255


def grid(bucket):
    w, h = INPUT[bucket]
    return h // PATCH, w // PATCH


def telemetry_features(telem, has):
    """z_mm, layer, progress, elapsed_s -> roughly unit-scale features."""
    z, layer, prog, el = telem.T
    f = np.stack([z / 100.0, np.log1p(np.maximum(layer, 0)) / 6.0, prog,
                  np.log1p(np.maximum(el, 0)) / 10.0], -1).astype(np.float32)
    return f * has[:, None]


class WindowDataset(Dataset):
    """decode='gpu': frames stay JPEG bytes for nvJPEG in the main process.
    decode='cpu': workers decode with libjpeg-turbo (cv2) and return uint8
    (4, 3, sh, sw) plus a validity mask, spreading the Huffman stage (which
    nvJPEG also runs on the CPU, serially) across the worker processes."""

    def __init__(self, rows, decode="gpu"):
        self.rows = rows
        self.decode = decode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if self.decode == "cpu":
            sw, sh = BUCKETS[r["bucket"]]
            frames = torch.zeros(4, 3, sh, sw, dtype=torch.uint8)
            valid = torch.zeros(4, dtype=torch.bool)
            for s, idx in enumerate(r["idx"]):
                if idx < 0:
                    continue
                img = cv2.imread(f"{r['path']}/frames/k{idx:03d}_v{r['var']}.jpg", cv2.IMREAD_COLOR_RGB)
                if img.shape[:2] != (sh, sw):
                    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
                frames[s] = torch.from_numpy(img).permute(2, 0, 1)
                valid[s] = True
            frames = (frames, valid)
        else:
            frames = []
            for idx in r["idx"]:
                if idx < 0:
                    frames.append(None)
                    continue
                with open(f"{r['path']}/frames/k{idx:03d}_v{r['var']}.jpg", "rb") as f:
                    frames.append(torch.frombuffer(bytearray(f.read()), dtype=torch.uint8))
        gh, gw = grid(r["bucket"])
        mask = cv2.imread(f"{r['path']}/masks/k{r['idx'][0]:03d}_v{r['var']}.png",
                          cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask[..., 0]
        onehot = np.stack([(mask == c) for c in TYPE_CLASS], -1).astype(np.float32)
        target = cv2.resize(onehot, (gw * TARGET_SUB, gh * TARGET_SUB), interpolation=cv2.INTER_AREA)
        return i, frames, torch.from_numpy(target.transpose(2, 0, 1).copy()).half()


def collate(items):
    ids = torch.tensor([it[0] for it in items])
    frames = [it[1] for it in items]
    if isinstance(frames[0], tuple):
        frames = (torch.stack([f[0] for f in frames]), torch.stack([f[1] for f in frames]))
    targets = torch.stack([it[2] for it in items])
    return ids, frames, targets


class BucketBatchSampler(Sampler):
    """Batches never mix buckets (one static shape per bucket for compile).
    `weights` (optional) gives per-sample draw probabilities; an epoch is
    then `len(rows)` weighted draws with replacement."""

    def __init__(self, buckets, batch_size, weights=None, seed=0, drop_last=True):
        self.buckets = np.asarray(buckets)
        self.batch_size = batch_size
        self.weights = None if weights is None else np.asarray(weights, np.float64)
        self.seed = seed
        self.epoch = 0
        self.skip = 0
        self.drop_last = drop_last

    def set_epoch(self, epoch, skip=0):
        """`skip` drops the first batches of the epoch (mid-epoch resume)."""
        self.epoch = epoch
        self.skip = skip

    def _batches(self):
        rng = np.random.default_rng(self.seed + 7919 * self.epoch)
        n = len(self.buckets)
        if self.weights is None:
            order = rng.permutation(n)
        else:
            order = rng.choice(n, size=n, replace=True, p=self.weights / self.weights.sum())
        batches = []
        for b in np.unique(self.buckets):
            sel = order[self.buckets[order] == b]
            k = len(sel) // self.batch_size if self.drop_last else math.ceil(len(sel) / self.batch_size)
            batches += [sel[j * self.batch_size:(j + 1) * self.batch_size] for j in range(k)]
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter([b.tolist() for b in self._batches()[self.skip:]])

    def __len__(self):
        return len(self._batches()) - self.skip


def decode_batch(frames, bucket, device):
    """List (B) of lists (4 slots) of JPEG bytes or None -> uint8 frames
    (B, 4, 3, H, W) at the stored bucket size, and a validity mask (B, 4)."""
    if isinstance(frames, tuple):  # decoded by the workers
        return frames[0].to(device, non_blocking=True), frames[1].to(device, non_blocking=True)
    sw, sh = BUCKETS[bucket]
    B = len(frames)
    out = torch.zeros(B, 4, 3, sh, sw, dtype=torch.uint8, device=device)
    valid = torch.zeros(B, 4, dtype=torch.bool)
    flat, where = [], []
    for b, slots in enumerate(frames):
        for s, data in enumerate(slots):
            if data is not None:
                flat.append(data)
                where.append((b, s))
    decoded = decode_jpeg(flat, device=device)
    for (b, s), img in zip(where, decoded):
        if img.shape[1:] != (sh, sw):
            img = F.interpolate(img[None].float(), size=(sh, sw), mode="bilinear",
                                antialias=True)[0].clamp(0, 255).byte()
        out[b, s] = img
        valid[b, s] = True
    return out, valid.to(device)


def gpu_augment(frames, valid, targets, train, bucket, gen=None):
    """frames uint8 (B,4,3,sh,sw) -> normalized float (B,4,3,H,W) at the
    model input size; targets (B,4,gh*S,gw*S) -> patch fractions (B,4,gh,gw).

    Train-time: one random zoom/crop + horizontal flip per window, applied
    identically to all its frames and to the target (fixed camera), and a
    mild per-window brightness/contrast jitter with small per-frame drift
    (auto-exposure)."""
    B = frames.shape[0]
    W, H = INPUT[bucket]
    gh, gw = grid(bucket)
    dev = frames.device
    # Antialiased resize first, so the zoom below only ever magnifies.
    x = F.interpolate(frames.flatten(0, 1).float(), size=(H, W), mode="bilinear", antialias=True)
    t = targets.float()
    if train:
        scale = torch.empty(B, device=dev).uniform_(0.8, 1.0, generator=gen)
        tx = (torch.rand(B, device=dev, generator=gen) * 2 - 1) * (1 - scale)
        ty = (torch.rand(B, device=dev, generator=gen) * 2 - 1) * (1 - scale)
        flip = torch.where(torch.rand(B, device=dev, generator=gen) < 0.5, -1.0, 1.0)
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0] = scale * flip
        theta[:, 0, 2] = tx
        theta[:, 1, 1] = scale
        theta[:, 1, 2] = ty
        g = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
        x = F.grid_sample(x, g.repeat_interleave(4, 0), mode="bilinear", padding_mode="border",
                          align_corners=False)
        gt = F.affine_grid(theta, (B, 1, gh * TARGET_SUB, gw * TARGET_SUB), align_corners=False)
        t = F.grid_sample(t, gt, mode="bilinear", align_corners=False)
        x = x.view(B, 4, 3, H, W)
        gain = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0.8, 1.2, generator=gen)
        gain = gain * torch.empty(B, 4, 1, 1, 1, device=dev).uniform_(0.95, 1.05, generator=gen)
        bias = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(-12, 12, generator=gen)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        x = ((x - mean) * gain + mean + bias).clamp(0, 255)
    else:
        x = x.view(B, 4, 3, H, W)
    x = (x - MEAN.to(dev)) / STD.to(dev)
    x = x * valid[:, :, None, None, None]
    patches = F.avg_pool2d(t, TARGET_SUB)
    return x, patches
