"""Sample index: one row per (job, window, ISP variant), plus real photos.

Scanning 40k `labels.json` files takes minutes, so each run's rows are cached
in `$CACHE/index/<run>.npz` together with the set of job ids they cover; a
rescan only reads jobs that finished since (main1 grows while we train).

Row fields (structured numpy array, see ROW):
    path      job directory (synthetic) or cache directory (real)
    var       ISP variant; frames are `frames/k{idx:03d}_v{var}.jpg`
    idx       keyframe index per slot, slot 0 = newest (t), then t-10, t-60,
              t-300; -1 where the window has no frame for that slot
    gap_s     newest t minus slot t (0 where missing)
    bucket    0 = 16:9 (448x252), 1 = 4:3 (448x336)
    presence  per type (TYPES), newest frame, this variant
    known     per type: 1 if presence is a usable target (0 for `ignore`
              and for types real photos do not label)
    area      visible class fraction per type, newest frame
    telem     z_mm, layer, progress, elapsed_s of the newest frame
    has_telem 0 for real photos
    split     train / val / test / test_setup / real_train / real_val / real_test
    source    synthetic setup name, or 'real_web' / 'real_webcam'
    ftype     the job's own failure type ('' for clean jobs / real)
"""

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import numpy as np

TYPES = ("spaghetti", "detached", "layer_shift", "nozzle_blob")
# IndexOB classes for TYPES (CONTRACTS.md §5).
TYPE_CLASS = (4, 5, 6, 7)
SLOT_GAPS = (0.0, 10.0, 60.0, 300.0)
BUCKETS = ((640, 360), (640, 480))  # stored frame size per bucket

SYNTH_ROOT = Path(os.environ.get("SCYTHE_SYNTH_ROOT", "/mnt/Primary/scythe-synth"))
CACHE = Path(os.environ.get("SCYTHE_TRAIN_CACHE", str(SYNTH_ROOT / "train_cache")))
REAL_ROOT = Path(__file__).resolve().parents[3] / "datasets" / "real_v1"

ROW = np.dtype([
    ("path", "U160"), ("var", "i1"), ("idx", "i2", 4), ("gap_s", "f4", 4), ("bucket", "i1"),
    ("presence", "i1", 4), ("known", "i1", 4), ("area", "f4", 4), ("telem", "f4", 4),
    ("has_telem", "i1"), ("split", "U12"), ("source", "U24"), ("ftype", "U12"),
])


def bucket_of(width, height):
    return 1 if abs(width / height - 4 / 3) < abs(width / height - 16 / 9) else 0


def done_jobs(run_dir):
    """Job ids the manifest marks done (read-only; the run may be live)."""
    db = run_dir / "manifest.sqlite"
    if not db.exists():
        return sorted(p.parent.name for p in run_dir.glob("jobs/*/labels.json"))
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    try:
        return sorted(r[0] for r in con.execute("SELECT job_id FROM jobs WHERE state='done'"))
    finally:
        con.close()


def job_rows(job_dir):
    labels = json.loads((job_dir / "labels.json").read_text())
    if not labels.get("alignment_ok", True):
        return []
    spec = labels["spec"]
    keys = labels["keyframes"]
    by_t = {round(k["t"], 3): k for k in keys}
    width, height = labels["variants"][0]["params"].get("resolution", BUCKETS[0])
    bucket = bucket_of(width, height)
    ftype = (spec.get("failure") or {}).get("type", "")
    rows = []
    for window in labels["windows"]:
        newest = by_t.get(round(window["t"], 3))
        if newest is None:
            continue
        idx = [newest["idx"], -1, -1, -1]
        gap = [0.0, 0.0, 0.0, 0.0]
        for h in window["history"]:
            k = by_t.get(round(h, 3))
            if k is None:
                continue
            d = newest["t"] - k["t"]
            slot = int(np.argmin([abs(d - g) for g in SLOT_GAPS[1:]])) + 1
            idx[slot], gap[slot] = k["idx"], d
        tel = newest["telemetry"]
        telem = [tel.get("z_mm", 0.0), tel.get("layer", 0), tel.get("progress", 0.0),
                 tel.get("elapsed_s", 0.0)]
        for var, v in enumerate(newest["variants"]):
            presence = [int(v["presence"][t]) for t in TYPES]
            known = [0 if v["ignore"][t] else 1 for t in TYPES]
            area = [float(v["area_frac"][t]) for t in TYPES]
            rows.append((str(job_dir), var, idx, gap, bucket, presence, known, area, telem, 1,
                         spec["split"], spec["setup"], ftype))
    return rows


def scan_run(run_dir, log=print):
    run_dir = Path(run_dir)
    cache = CACHE / "index" / f"{run_dir.name}.npz"
    rows, seen = np.zeros(0, ROW), set()
    if cache.exists():
        old = np.load(cache, allow_pickle=False)
        rows, seen = old["rows"], set(old["jobs"].tolist())
    todo = [j for j in done_jobs(run_dir) if j not in seen]
    new = []
    for i, job in enumerate(todo):
        try:
            new += job_rows(run_dir / "jobs" / job)
        except (OSError, KeyError, ValueError) as exc:
            log(f"skip {job}: {exc}")
            continue
        seen.add(job)
        if i and i % 2000 == 0:
            log(f"  {run_dir.name}: {i}/{len(todo)} jobs")
    if todo:
        rows = np.concatenate([rows, np.array(new, ROW)])
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp.npz")
        np.savez(tmp, rows=rows, jobs=np.array(sorted(seen)))
        os.replace(tmp, cache)
    log(f"{run_dir.name}: {len(seen)} jobs, {len(rows)} samples (+{len(todo)} jobs)")
    return rows


def build_real(log=print):
    """Real photos -> cache of letterboxed frames at a stored bucket size,
    with a class-4 mask from the spaghetti boxes, so they load exactly like
    synthetic single-frame windows. Only spaghetti is labelled; the other
    types are unknown (not negative), except for the user's own webcam
    frames, which are clean."""
    import cv2

    out = CACHE / "real_v2"
    cache = out / "rows.npy"
    if cache.exists():
        return np.load(cache)
    rows = []
    for line in (REAL_ROOT / "manifest.jsonl").read_text().splitlines():
        rec = json.loads(line)
        img = cv2.imread(str(REAL_ROOT / rec["path"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        bucket = bucket_of(w, h)
        bw, bh = BUCKETS[bucket]
        s = min(bw / w, bh / h)
        nw, nh = round(w * s), round(h * s)
        x0, y0 = (bw - nw) // 2, (bh - nh) // 2
        canvas = np.zeros((bh, bw, 3), np.uint8)
        canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        mask = np.ones((bh, bw), np.uint8)
        label = (REAL_ROOT / rec["path"].replace("images/", "labels/")).with_suffix(".txt")
        if label.exists():
            for parts in (ln.split() for ln in label.read_text().splitlines() if ln.strip()):
                if int(parts[0]) != 1:  # 0 = printer-gantry
                    continue
                cx, cy, bw_, bh_ = map(float, parts[1:5])
                xa, xb = x0 + (cx - bw_ / 2) * nw, x0 + (cx + bw_ / 2) * nw
                ya, yb = y0 + (cy - bh_ / 2) * nh, y0 + (cy + bh_ / 2) * nh
                mask[int(max(0, ya)):int(np.ceil(min(bh, yb))),
                     int(max(0, xa)):int(np.ceil(min(bw, xb)))] = 4
        # Roboflow stems only differ after ~60 chars; keep them unique.
        stem = Path(rec["path"]).stem
        name = stem[:40] + "_" + hashlib.sha1(stem.encode()).hexdigest()[:12]
        d = out / rec["split"] / name
        (d / "frames").mkdir(parents=True, exist_ok=True)
        (d / "masks").mkdir(exist_ok=True)
        cv2.imwrite(str(d / "frames" / "k000_v0.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(d / "masks" / "k000_v0.png"), mask)
        webcam = rec.get("source_type") == "personal_webcam" or "personal" in rec["path"]
        spag = int(rec.get("has_spaghetti", False))
        area = float((mask == 4).mean())
        known = [1, 1, 1, 1] if webcam else [1, 0, 0, 0]
        rows.append((str(d), 0, [0, -1, -1, -1], [0, 0, 0, 0], bucket, [spag, 0, 0, 0], known,
                     [area, 0, 0, 0], [0, 0, 0, 0], 0, "real_" + rec["split"],
                     "real_webcam" if webcam else "real_web", ""))
    rows = np.array(rows, ROW)
    np.save(cache, rows)
    log(f"real_v1: {len(rows)} photos ({int(rows['presence'][:, 0].sum())} with spaghetti)")
    return rows


def load_index(runs, real=True, log=print):
    parts = [scan_run(SYNTH_ROOT / "runs" / r, log) for r in runs]
    if real:
        parts.append(build_real(log))
    return np.concatenate(parts) if parts else np.zeros(0, ROW)
