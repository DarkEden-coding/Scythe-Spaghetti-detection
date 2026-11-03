"""Losses and metrics. Targets marked unknown (`known == 0`: generator
`ignore`, or types real photos don't label) never contribute."""

import numpy as np
import torch
import torch.nn.functional as F


def focal_bce(logits, target, mask, gamma=2.0, alpha=0.5):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    w = (alpha * target + (1 - alpha) * (1 - target)) * (1 - pt) ** gamma
    return (ce * w * mask).sum() / mask.sum().clamp_min(1.0)


def heatmap_loss(logits, frac, mask, pos_weight=8.0):
    """Patch logits vs. soft targets (visible class fraction per patch,
    saturating at 25% coverage), plus a soft dice term per (sample, type) so
    tiny failures are not drowned by background patches."""
    target = (frac * 4).clamp(0, 1)
    m = mask[:, :, None, None]
    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device))
    bce = (bce * m).sum() / (m.sum() * logits.shape[-1] * logits.shape[-2]).clamp_min(1.0)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((2, 3))
    denom = p.sum((2, 3)) + target.sum((2, 3))
    has = (target.sum((2, 3)) > 0).float() * mask
    dice = 1 - (2 * inter + 1) / (denom + 1)
    dice = (dice * has).sum() / has.sum().clamp_min(1.0)
    return bce + 0.5 * dice


def average_precision(score, label):
    order = np.argsort(-score, kind="stable")
    label = label[order]
    if label.sum() == 0:
        return float("nan")
    tp = np.cumsum(label)
    precision = tp / np.arange(1, len(label) + 1)
    return float((precision * label).sum() / label.sum())


def auroc(score, label):
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.r_[pos, neg]
    # Average ranks, so tied positive/negative pairs count 1/2.
    uniq, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recall_at_fpr(score, label, fpr=0.001):
    """Recall at the threshold where at most `fpr` of negatives fire:
    the operating regime that matters for a ~1 false pause / 100 h target."""
    neg, pos = score[label == 0], score[label == 1]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    thr = np.quantile(neg, 1 - fpr) if len(neg) * fpr >= 1 else neg.max()
    return float((pos > thr).mean()), float(thr)
