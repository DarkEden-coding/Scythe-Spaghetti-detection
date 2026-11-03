"""scythe-train {index,train,bench}."""

import argparse
import dataclasses
import json
import os
import time

# Batch sizes near the 8 GB limit thrash the allocator without this.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .train import Config


def add_config_args(p):
    for f in dataclasses.fields(Config):
        if f.type in (bool, "bool"):
            p.add_argument(f"--{f.name.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"),
                           default=f.default)
        elif f.name == "runs":
            p.add_argument("--runs", nargs="+", default=list(f.default))
        else:
            typ = {"int": int, "float": float, "str": str}.get(getattr(f.type, "__name__", f.type), str)
            p.add_argument(f"--{f.name.replace('_', '-')}", type=typ, default=f.default)


def config_from(args):
    return Config(**{f.name: (tuple(getattr(args, f.name)) if f.name == "runs" else getattr(args, f.name))
                     for f in dataclasses.fields(Config)})


def cmd_index(args):
    from .index import load_index
    rows = load_index(args.runs, real=True)
    import numpy as np
    for split in np.unique(rows["split"]):
        sel = rows[rows["split"] == split]
        print(f"{split:12s} {len(sel):7d} samples, positives per type {sel['presence'].sum(0).tolist()}")


def cmd_train(args):
    from .train import train
    train(config_from(args), log=lambda m: print(m, flush=True))


def cmd_bench(args):
    """GPU ceiling on random inputs for each (batch, frozen, compile) combo,
    then the real loader alone (no model) to see whether data keeps up."""
    import torch
    from .model import TemporalFailureNet
    from .data import INPUT
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    results = []
    for frozen in (args.frozen if args.batch else []):
        for ckpt in args.grad_checkpoint:
            model = TemporalFailureNet(frozen_blocks=frozen, grad_checkpoint=ckpt).to(dev)
            opt = torch.optim.AdamW(model.param_groups(3e-4), fused=True)
            for comp in args.compile:
                fwd = torch.compile(model, mode=args.compile_mode) if comp else model
                for bs in args.batch:
                    W, H = INPUT[0]
                    x = torch.randn(bs, 4, 3, H, W, device=dev)
                    valid = torch.ones(bs, 4, dtype=torch.bool, device=dev)
                    gaps = torch.tensor([0, 10, 60, 300.0], device=dev).expand(bs, 4).contiguous()
                    tel = torch.zeros(bs, 4, device=dev)
                    has = torch.ones(bs, device=dev)
                    torch.cuda.reset_peak_memory_stats()
                    try:
                        times = []
                        for i in range(args.iters + 5):
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                lo, he = fwd(x, valid, gaps, tel, has)
                            loss = lo.float().mean() + he.float().mean()
                            opt.zero_grad(set_to_none=True)
                            loss.backward()
                            opt.step()
                            torch.cuda.synchronize()
                            if i >= 5:
                                times.append(time.perf_counter() - t0)
                        r = {"frozen": frozen, "ckpt": ckpt, "compile": comp, "batch": bs,
                             "samples_s": round(bs / (sum(times) / len(times)), 1),
                             "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                    except torch.OutOfMemoryError:
                        r = {"frozen": frozen, "ckpt": ckpt, "compile": comp, "batch": bs, "oom": True}
                    # A failed step's activations stay referenced by these
                    # locals; drop them or every later config OOMs too.
                    lo = he = loss = x = None
                    opt.zero_grad(set_to_none=True)
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(json.dumps(r), flush=True)
                    results.append(r)
            del model, opt
            torch.cuda.empty_cache()
    if args.loader:
        from .index import load_index
        from .train import Batch, make_loader, sample_weights, split_rows
        cfg = config_from(args)
        rows, _ = split_rows(load_index(cfg.runs, real=cfg.real))
        for workers in args.loader_workers:
            cfg = dataclasses.replace(cfg, workers=workers)
            loader, _ = make_loader(rows, cfg, True, sample_weights(rows, cfg.real_share))
            prep = Batch(rows, dev)
            gen = torch.Generator(device=dev)
            n, t0 = 0, None
            for i, item in enumerate(loader):
                prep(item, train=True, cfg=cfg, gen=gen)
                if i == 10:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                elif i > 10:
                    n += cfg.batch_size
                if i >= 10 + args.iters * 2:
                    break
            torch.cuda.synchronize()
            r = {"loader_workers": workers, "batch": cfg.batch_size,
                 "samples_s": round(n / (time.perf_counter() - t0), 1)}
            print(json.dumps(r), flush=True)
            del loader


def main():
    p = argparse.ArgumentParser(prog="scythe-train")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index")
    pi.add_argument("--runs", nargs="+", default=list(Config.runs))
    pt = sub.add_parser("train")
    add_config_args(pt)
    pb = sub.add_parser("bench", conflict_handler="resolve")
    add_config_args(pb)
    pb.add_argument("--batch", type=int, nargs="*", default=[16, 24, 32])
    pb.add_argument("--frozen", type=int, nargs="+", default=[6])
    pb.add_argument("--grad-checkpoint", type=lambda s: s.lower() == "true", nargs="+", default=[False])
    pb.add_argument("--compile", type=lambda s: s.lower() == "true", nargs="+", default=[True])
    pb.add_argument("--iters", type=int, default=20)
    pb.add_argument("--loader", action="store_true")
    pb.add_argument("--loader-workers", type=int, nargs="+", default=[6])
    args = p.parse_args()
    {"index": cmd_index, "train": cmd_train, "bench": cmd_bench}[args.cmd](args)


if __name__ == "__main__":
    main()
