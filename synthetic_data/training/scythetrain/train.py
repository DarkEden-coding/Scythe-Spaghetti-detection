"""Training loop: bf16 autocast, torch.compile, fused AdamW, bucketed
batches, GPU decode/augment, atomic resumable checkpoints.

Output: $SYNTH_ROOT/train_runs/<name>/{config.json, log.jsonl, last.pt,
best.pt, eval_*.json}. `touch <dir>/STOP` stops cleanly after the current
step (checkpoint written); rerunning the same command resumes.
"""

import dataclasses
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (BucketBatchSampler, WindowDataset, collate, decode_batch, gpu_augment,
                   telemetry_features)
from .index import SYNTH_ROOT, TYPES, load_index
from .losses import auroc, average_precision, focal_bce, heatmap_loss, recall_at_fpr
from .model import TemporalFailureNet


@dataclasses.dataclass
class Config:
    name: str = "dev"
    runs: tuple = ("pilot2", "smoke3", "main1")
    real: bool = True
    real_share: float = 0.05        # fraction of training draws from real photos
    encoder: str = "vit_small_patch14_reg4_dinov2.lvd142m"
    frozen_blocks: int = 6
    fusion_blocks: int = 2
    grad_checkpoint: bool = False
    history_grad: bool = False      # backprop into the encoder through history frames too
    batch_size: int = 24
    lr: float = 3e-4
    encoder_lr_scale: float = 0.1
    layer_decay: float = 0.8
    weight_decay: float = 0.05
    warmup_steps: int = 300
    epochs: float = 3.0
    max_minutes: float = 0.0        # 0 = no limit
    heat_weight: float = 1.0
    history_drop: float = 0.15      # whole history dropped (single-frame mode)
    slot_drop: float = 0.1          # each history slot dropped independently
    telem_drop: float = 0.2         # also hides "no telemetry" as a real-photo tell
    workers: int = 6
    decode: str = "cpu"             # cpu (workers, libjpeg-turbo) | gpu (nvJPEG, main process)
    prefetch: int = 4
    compile: bool = True
    compile_mode: str = "max-autotune-no-cudagraphs"
    eval_every_min: float = 30.0
    ckpt_every_min: float = 10.0
    log_every: int = 25
    max_eval: int = 6000
    seed: int = 0
    # Cap on this process's share of GPU memory (0 = none). Training next to
    # the render run without one starves Blender (OptiX/OOM job failures);
    # use e.g. --batch-size 16 --mem-fraction 0.55 there.
    mem_fraction: float = 0.0


def split_rows(rows):
    train = np.isin(rows["split"], ["train", "real_train"])
    val = np.isin(rows["split"], ["val", "real_val"])
    return rows[train], rows[val]


def sample_weights(rows, real_share):
    real = np.char.startswith(rows["source"], "real")
    w = np.ones(len(rows))
    if real.any() and (~real).any():
        w[real] = real_share / real.sum()
        w[~real] = (1 - real_share) / (~real).sum()
    return w


class LossModel(torch.nn.Module):
    """Model + losses in one module, so torch.compile fuses the loss math
    into the training graph."""

    def __init__(self, model, heat_weight):
        super().__init__()
        self.model, self.heat_weight = model, heat_weight

    def forward(self, x, valid, gaps, telem, has, presence, known, heat_known, frac):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, heat = self.model(x, valid, gaps, telem, has)
        l_cls = focal_bce(logits.float(), presence, known)
        l_heat = heatmap_loss(heat.float(), frac, heat_known)
        return l_cls + self.heat_weight * l_heat, l_cls, l_heat


class Batch:
    """Moves one collated batch to the GPU and builds model inputs/targets."""

    def __init__(self, rows, device):
        self.rows, self.device = rows, device

    def __call__(self, item, train, cfg=None, gen=None):
        ids, frames, targets = item
        r = self.rows[ids.numpy()]
        bucket = int(r["bucket"][0])
        dev = self.device
        raw, valid = decode_batch(frames, bucket, dev)
        x, frac = gpu_augment(raw, valid, targets.to(dev, non_blocking=True), train, bucket, gen)
        gaps = torch.from_numpy(np.ascontiguousarray(r["gap_s"])).to(dev)
        has = torch.from_numpy(r["has_telem"].astype(np.float32)).to(dev)
        telem = torch.from_numpy(telemetry_features(r["telem"], r["has_telem"])).to(dev)
        if train:
            B = len(r)
            drop_all = torch.rand(B, 1, device=dev, generator=gen) < cfg.history_drop
            drop_one = torch.rand(B, 4, device=dev, generator=gen) < cfg.slot_drop
            keep = ~(drop_all | drop_one)
            keep[:, 0] = True
            valid = valid & keep
            x = x * valid[:, :, None, None, None]
            tdrop = (torch.rand(B, device=dev, generator=gen) < cfg.telem_drop).float()
            telem = telem * (1 - tdrop[:, None])
            has = has * (1 - tdrop)
        gaps = gaps * valid
        presence = torch.from_numpy(np.ascontiguousarray(r["presence"], np.float32)).to(dev)
        known = torch.from_numpy(np.ascontiguousarray(r["known"], np.float32)).to(dev)
        # Web-photo boxes are mostly background pixels: presence labels only,
        # no heatmap target.
        web = torch.from_numpy(r["source"] == "real_web").to(dev)
        heat_known = known * ~web[:, None]
        return dict(x=x, valid=valid, gaps=gaps, telem=telem, has=has, presence=presence,
                    known=known, heat_known=heat_known, frac=frac, ids=ids)


def lr_at(step, total, cfg):
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    p = min(1.0, (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps))
    return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * p))


def save_atomic(obj, path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, loader, prep, rows, log_dir=None, step=0):
    model.eval()
    scores, labels, known, srcs = [], [], [], []
    for item in loader:
        b = prep(item, train=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(b["x"], b["valid"], b["gaps"], b["telem"], b["has"])
        # Rank on fp32 logits: sigmoid saturates to exactly 1.0 and ties.
        scores.append(logits.float().cpu().numpy())
        labels.append(b["presence"].cpu().numpy())
        known.append(b["known"].cpu().numpy())
        srcs.append(rows["source"][b["ids"].numpy()])
    model.train()
    s, y, k, src = map(np.concatenate, (scores, labels, known, srcs))
    real = np.char.startswith(src, "real")
    out = {"step": step, "n": int(len(s))}
    for domain, sel in (("synth", ~real), ("real", real)):
        for j, t in enumerate(TYPES):
            m = sel & (k[:, j] > 0)
            if m.sum() < 10 or y[m, j].sum() == 0:
                continue
            rec, thr = recall_at_fpr(s[m, j], y[m, j], 0.001)
            out[f"{domain}/{t}"] = {"ap": round(average_precision(s[m, j], y[m, j]), 4),
                                    "auroc": round(auroc(s[m, j], y[m, j]), 4),
                                    "recall@fpr1e-3": round(rec, 4), "n_pos": int(y[m, j].sum()),
                                    "n": int(m.sum())}
    aps = [v["ap"] for kk, v in out.items() if kk.startswith("synth/") and not np.isnan(v["ap"])]
    out["synth_map"] = round(float(np.mean(aps)), 4) if aps else float("nan")
    if log_dir:
        (Path(log_dir) / f"eval_{step:07d}.json").write_text(json.dumps(out, indent=1))
    return out


def make_loader(rows, cfg, train, weights=None):
    sampler = BucketBatchSampler(rows["bucket"], cfg.batch_size, weights=weights,
                                 seed=cfg.seed, drop_last=train)
    workers = cfg.workers if train else min(cfg.workers, 3)
    loader = DataLoader(WindowDataset(rows, cfg.decode), batch_sampler=sampler, collate_fn=collate,
                        pin_memory=cfg.decode == "cpu",
                        num_workers=workers, prefetch_factor=cfg.prefetch if workers else None,
                        persistent_workers=train and workers > 0)
    return loader, sampler


def train(cfg: Config, log=print):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed)
    if cfg.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(cfg.mem_fraction)
    out = SYNTH_ROOT / "train_runs" / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "STOP").unlink(missing_ok=True)
    snapshot = out / "rows.npz"
    if snapshot.exists():
        # The index keeps growing while generation runs; a resumed run must
        # see exactly the rows (and so batch order and LR schedule) it began with.
        snap = np.load(snapshot)
        train_rows, val_rows = snap["train"], snap["val"]
    else:
        (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
        train_rows, val_rows = split_rows(load_index(cfg.runs, real=cfg.real, log=log))
        real = np.char.startswith(val_rows["source"], "real")
        synth = val_rows[~real]
        if len(synth) > cfg.max_eval:  # cap synthetic only; real val is tiny, keep all
            synth = synth[np.random.default_rng(0).choice(len(synth), cfg.max_eval, replace=False)]
        val_rows = np.concatenate([synth, val_rows[real]])
        np.savez(out / "rows.tmp.npz", train=train_rows, val=val_rows)
        os.replace(out / "rows.tmp.npz", snapshot)
    log(f"train {len(train_rows)} samples, val {len(val_rows)}")
    device = torch.device("cuda")
    model = TemporalFailureNet(cfg.encoder, cfg.frozen_blocks, cfg.fusion_blocks,
                               grad_checkpoint=cfg.grad_checkpoint,
                               history_grad=cfg.history_grad).to(device)
    opt = torch.optim.AdamW(model.param_groups(cfg.lr, cfg.encoder_lr_scale, cfg.layer_decay,
                                               cfg.weight_decay), lr=cfg.lr, fused=True)
    loader, sampler = make_loader(train_rows, cfg, True, sample_weights(train_rows, cfg.real_share))
    vloader, _ = make_loader(val_rows, cfg, False)
    steps_per_epoch = len(sampler._batches())
    total = int(cfg.epochs * steps_per_epoch)
    step, epoch, epoch_step, best = 0, 0, 0, -1.0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, epoch, best = ck["step"], ck["epoch"], ck.get("best", -1.0)
        epoch_step = ck.get("epoch_step", 0)
        log(f"resumed at step {step} (epoch {epoch}, batch {epoch_step})")
    # Evaluation runs eager: it is short, and its ragged last batch and
    # eval-mode path would otherwise each cost a max-autotune compile.
    fwd = model
    train_fwd = LossModel(model, cfg.heat_weight)
    if cfg.compile:
        # dynamic=False: one static graph per bucket, never a dynamic-shape one.
        train_fwd = torch.compile(train_fwd, mode=cfg.compile_mode, dynamic=False)
    prep = Batch(train_rows, device)
    vprep = Batch(val_rows, device)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed * 1_000_003 + step)

    def checkpoint(path):
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                     "epoch": epoch, "epoch_step": epoch_step, "best": best,
                     "config": dataclasses.asdict(cfg)}, path)

    t_start = time.time()
    last_ckpt = last_eval = time.time()
    logf = open(out / "log.jsonl", "a")
    stop = False
    bad_steps = 0
    bad_gpu = torch.zeros((), dtype=torch.long, device=device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    first = True
    while step < total and not stop:
        sampler.set_epoch(epoch, skip=epoch_step)  # resume mid-epoch
        meter = {"loss": 0.0, "cls": 0.0, "heat": 0.0, "n": 0, "data": 0.0, "t0": time.time()}
        t_data = time.time()
        for item in loader:
            b = prep(item, train=True, cfg=cfg, gen=gen)
            meter["data"] += time.time() - t_data
            for g in opt.param_groups:
                g["lr"] = cfg.lr * g["lr_scale"] * lr_at(step, total, cfg)
            loss, l_cls, l_heat = train_fwd(b["x"], b["valid"], b["gaps"], b["telem"], b["has"],
                                            b["presence"], b["known"], b["heat_known"], b["frac"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            # One non-finite update poisons every weight and Adam state. Fused
            # AdamW skips the whole update (moments, decay, step count) when
            # found_inf is set, decided on the GPU without a host sync; the
            # count is read at log steps.
            bad = ~torch.isfinite(gnorm + loss)
            bad_gpu += bad
            opt.found_inf = bad.float()
            opt.step()
            step += 1
            epoch_step += 1
            meter["n"] += 1
            if step % cfg.log_every == 0:
                n_bad = int(bad_gpu.item())
                if n_bad > bad_steps:
                    log(json.dumps({"step": step, "bad_steps": n_bad}))
                    logf.write(json.dumps({"step": step, "bad_steps": n_bad}) + "\n")
                    bad_steps = n_bad
                    if n_bad > 20:
                        raise RuntimeError("too many non-finite steps; see log.jsonl")
                meter["loss"] += loss.item()
                meter["cls"] += l_cls.item()
                meter["heat"] += l_heat.item()
                dt = time.time() - meter["t0"]
                rec = {"step": step, "epoch": round(step / steps_per_epoch, 3),
                       "loss": round(meter["loss"], 4), "cls": round(meter["cls"], 4),
                       "heat": round(meter["heat"], 4),
                       "samples_s": round(meter["n"] * cfg.batch_size / dt, 1),
                       "data_frac": round(meter["data"] / dt, 3),
                       "lr": opt.param_groups[0]["lr"],
                       "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                log(json.dumps(rec))
                meter = {"loss": 0.0, "cls": 0.0, "heat": 0.0, "n": 0, "data": 0.0, "t0": time.time()}
            now = time.time()
            if first:  # compile time does not count towards the eval/ckpt timers
                last_eval = last_ckpt = now
                first = False
            if (out / "STOP").exists():
                stop = True
            if cfg.max_minutes and now - t_start > cfg.max_minutes * 60:
                stop = True
            if stop or now - last_ckpt > cfg.ckpt_every_min * 60:
                checkpoint(out / "last.pt")
                last_ckpt = now
            if not stop and now - last_eval > cfg.eval_every_min * 60:
                res = evaluate(fwd, vloader, vprep, val_rows, out, step)
                log(json.dumps(res))
                if res["synth_map"] > best:
                    best = res["synth_map"]
                    save_atomic({"model": model.state_dict(), "step": step,
                                 "config": dataclasses.asdict(cfg)}, out / "best.pt")
                last_eval = time.time()
            if stop or step >= total:
                break
            t_data = time.time()
        else:
            epoch += 1
            epoch_step = 0
    checkpoint(out / "last.pt")
    res = evaluate(fwd, vloader, vprep, val_rows, out, step)
    log(json.dumps(res))
    if res["synth_map"] > best:
        save_atomic({"model": model.state_dict(), "step": step, "config": dataclasses.asdict(cfg)},
                    out / "best.pt")
    logf.close()
    return res
