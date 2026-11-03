"""Download Thingi10K once and index printable, permissively licensed meshes.

The pinned NPZ variant preserves source coordinates. No scaling is done here.
All package/download caches are redirected below corpus/raw before importing HF.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import trimesh

from .paths import CORPUS_DIR

FILTER_VERSION = 1


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, allow_nan=False)
        name = f.name
    os.replace(name, path)


def permissive_license(value: str | None) -> str | None:
    """Conservative allowlist; preserve the original license for attribution."""
    if not value:
        return None
    text = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    tokens = set(text.split())
    if (tokens & {"nc", "nd", "noncommercial", "noderivatives", "unknown"}
            or "non commercial" in text or "no derivatives" in text
            or "no derivative" in text):
        return None
    if ("public domain" in text or "cc0" in tokens or "cc 0" in text
            or "gpl" in tokens or "lgpl" in tokens or text == "bsd license"
            or "gnu general public" in text or "gnu lesser general public" in text):
        return value
    if re.fullmatch(r"cc by(?: sa)?(?: \d+(?: \d+)?)?", text):
        return value
    if text.startswith("creative commons attribution"):
        tail = text.removeprefix("creative commons attribution").strip()
        if re.fullmatch(r"(?:share ?alike)?(?: \d+(?: \d+)?)?(?: international| unported)?", tail):
            return value
    return None


def metadata_rejection(entry: dict) -> str | None:
    if not permissive_license(entry.get("license")):
        return "license"
    if not all(entry.get(k) for k in ("closed", "vertex_manifold", "edge_manifold", "solid")):
        return "not_closed_manifold_solid"
    if entry.get("num_components") != 1:
        return "multiple_bodies"
    if entry.get("self_intersecting"):
        return "self_intersection"
    if not 4 <= entry.get("num_facets", 0) <= 500_000:
        return "face_count"
    return None


def inspect_mesh(mesh: trimesh.Trimesh) -> dict:
    """Validate source geometry, returning unscaled dimensions and overhang area.

    Overhangs exclude the bottom 0.1% of Z (the build-plate contact surface).
    The threshold is 1% of total surface area at >45 degrees facing downward.
    Allowed uniform slice-time scaling is 0.001–100x, with a 100 mm height cap.
    """
    if not 4 <= len(mesh.faces) <= 500_000 or not np.isfinite(mesh.vertices).all():
        raise ValueError("face_count_or_nonfinite")
    size = mesh.extents
    if np.any(size <= 1e-6) or max(size) / min(size) > 80:
        raise ValueError("degenerate_or_aspect_ratio")
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.body_count != 1:
        raise ValueError("not_closed_single_body")
    if mesh.volume <= 1e-12 or np.any(mesh.area_faces <= 1e-14):
        raise ValueError("degenerate_faces_or_volume")
    # A feasible printable scale on the smallest plate, with room for a brim.
    scale = min(153 / max(size[:2]), 100 / size[2], 100)
    if scale < 0.001 or min(size * scale) < 0.4:
        raise ValueError("unprintable_size")
    above_base = mesh.triangles_center[:, 2] > mesh.bounds[0, 2] + size[2] * 0.001
    downward = (mesh.face_normals[:, 2] < -np.sqrt(0.5)) & above_base
    fraction = float(mesh.area_faces[downward].sum() / mesh.area)
    return {"faces": len(mesh.faces), "bbox_mm": size.tolist(),
            "overhang_fraction": fraction, "has_overhangs": fraction >= 0.01}


def download_metadata(raw: Path) -> list[dict]:
    raw.mkdir(parents=True, exist_ok=True)
    for key, subdir in (("HF_HOME", "huggingface"), ("HF_DATASETS_CACHE", "datasets"),
                        ("HF_HUB_CACHE", "hub"), ("XDG_CACHE_HOME", "cache"),
                        ("TMPDIR", "tmp")):
        os.environ[key] = str(raw / subdir)
        (raw / subdir).mkdir(exist_ok=True)
    metadata = raw / "metadata.json"
    if metadata.exists():
        rows = json.loads(metadata.read_text())
        if rows and all(Path(row["file_path"]).is_file() for row in rows):
            return rows
    import thingi10k
    thingi10k.init(cache_dir=str(raw), variant="npz")
    rows = list(thingi10k.dataset())
    atomic_json(metadata, rows)
    return rows


def build_corpus(*, limit: int | None = None, min_overhang: float = 0,
                 max_overhang: float = 1, corpus_dir: Path | None = None) -> list[dict]:
    root = Path(corpus_dir or CORPUS_DIR).resolve()
    rows = download_metadata(root / "raw")
    stls = root / "stl"
    stls.mkdir(parents=True, exist_ok=True)
    cache_path = root / "raw" / "inspection.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    accepted, rejected = [], Counter()
    for n, row in enumerate(sorted(rows, key=lambda r: int(r["file_id"]))):
        reason = metadata_rejection(row)
        if reason:
            rejected[reason] += 1
            continue
        model_id = f"thingi10k_{row['file_id']}"
        src = Path(row["file_path"])
        signature = [FILTER_VERSION, src.stat().st_size, src.stat().st_mtime_ns]
        saved = cache.get(model_id, {})
        try:
            if saved.get("signature") == signature and saved.get("rejection"):
                rejected[saved["rejection"]] += 1
                continue
            mesh = None
            if saved.get("signature") == signature:
                info = saved["info"]
            else:
                with np.load(src, allow_pickle=False) as data:
                    mesh = trimesh.Trimesh(vertices=data["vertices"], faces=data["facets"],
                                           process=True)
                info = inspect_mesh(mesh)
                cache[model_id] = {"signature": signature, "info": info}
            if not min_overhang <= info["overhang_fraction"] <= max_overhang:
                rejected["overhang_range"] += 1
                continue
            dest = stls / f"{model_id}.stl"
            if not dest.exists():
                if mesh is None:
                    with np.load(src, allow_pickle=False) as data:
                        mesh = trimesh.Trimesh(vertices=data["vertices"], faces=data["facets"],
                                               process=True)
                tmp = dest.with_suffix(".stl.tmp")
                tmp.write_bytes(mesh.export(file_type="stl"))
                tmp.replace(dest)
            accepted.append({"id": model_id, "stl": str(dest), "license": row["license"],
                             **info, **{k: row.get(k) for k in
                                       ("name", "author", "thing_id", "tags", "category", "subcategory")}})
        except (ValueError, OSError, KeyError) as exc:
            reason = str(exc)
            rejected[reason] += 1
            cache[model_id] = {"signature": signature, "rejection": reason}
        if n % 100 == 0:
            atomic_json(cache_path, cache)
            atomic_json(root / "index.json", accepted)
            print(f"Inspected {n + 1}/{len(rows)}; accepted {len(accepted)}", flush=True)
        if limit is not None and len(accepted) >= limit:
            break
    atomic_json(cache_path, cache)
    atomic_json(root / "index.json", accepted)
    stats = {"downloaded": len(rows), "accepted": len(accepted), "rejected": dict(rejected),
             "licenses": dict(Counter(e["license"] for e in accepted)),
             "with_overhangs": sum(e["has_overhangs"] for e in accepted),
             "filter_version": FILTER_VERSION, "limit": limit,
             "min_overhang": min_overhang, "max_overhang": max_overhang}
    atomic_json(root / "stats.json", stats)
    print(json.dumps(stats, indent=2))
    return accepted


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-overhang", type=float, default=0,
                        help="Minimum downward overhang area / total area, excluding base")
    parser.add_argument("--max-overhang", type=float, default=1)


def run(args) -> int:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if not 0 <= args.min_overhang <= args.max_overhang <= 1:
        raise ValueError("overhang fractions must satisfy 0 <= min <= max <= 1")
    build_corpus(limit=args.limit, min_overhang=args.min_overhang, max_overhang=args.max_overhang)
    return 0
