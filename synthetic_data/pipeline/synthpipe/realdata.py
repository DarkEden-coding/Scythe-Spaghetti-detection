"""Rebuild real YOLO splits with auditable provenance and perceptual grouping."""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .paths import REAL_DATASETS

SOURCES = ("3d-printing-fail-detection-v10", "personal_printer_augmentation-v1")
SPLITS = ("train", "val", "test")


def original_stem(filename):
    return re.sub(
        r"_(?:jpg|jpeg|png|webp|bmp)$", "", filename.split(".rf.")[0], flags=re.IGNORECASE
    )


def video_key(stem):
    match = re.match(r"(.*(?:MOV|MP4|AVI))[-_]\d+$", stem, re.IGNORECASE)
    return match.group(1) if match else None


@dataclass
class Record:
    path: str
    label: str
    dataset: str
    old_split: str
    original: str
    source_type: str
    width: int
    height: int
    boxes: list[int]
    phash: list[int]
    dhash: list[int]
    color: float
    sharpness: float
    group: str = ""
    split: str = ""
    kept: bool = True
    polygon_boxes: int = 0
    removed_reason: str = ""


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def _bits(values):
    return int.from_bytes(np.packbits(values.ravel()).tobytes(), "big")


def image_features(path):
    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
        width, height = rgb.size
        small = np.asarray(rgb.resize((128, 128)), dtype=np.float32)
        color = float(np.mean(np.std(small, axis=2)))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        ph, dh = [], []
        for fraction in (1.0, 0.9, 0.8):
            dx, dy = width * (1 - fraction) / 2, height * (1 - fraction) / 2
            crop = rgb.crop((round(dx), round(dy), round(width - dx), round(height - dy)))
            for view in (crop, ImageOps.mirror(crop)):
                g = view.convert("L")
                dct = cv2.dct(np.asarray(g.resize((32, 32), Image.Resampling.LANCZOS), np.float32))[
                    :8, :8
                ]
                ph.append(_bits(dct > np.median(dct.ravel()[1:])))
                thumb = np.asarray(g.resize((9, 8), Image.Resampling.LANCZOS))
                dh.append(_bits(thumb[:, 1:] > thumb[:, :-1]))
    return width, height, ph, dh, color, sharpness


def normalized_labels(path):
    """Preserve box rows verbatim; convert occasional segmentation polygons to XYWH."""
    boxes, rows, polygons = [], [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 5 or len(fields) % 2 != 1 or int(fields[0]) not in (0, 1):
            raise ValueError(f"Invalid annotation: {path}: {line}")
        coords = np.asarray([float(x) for x in fields[1:]])
        if not np.all(np.isfinite(coords) & (coords >= 0) & (coords <= 1)):
            raise ValueError(f"Invalid YOLO coordinates: {path}")
        boxes.append(int(fields[0]))
        if len(fields) > 5:
            points = coords.reshape(-1, 2)
            low, high = points.min(axis=0), points.max(axis=0)
            center, size = (low + high) / 2, high - low
            line = fields[0] + " " + " ".join(f"{v:.12g}" for v in (*center, *size))
            polygons += 1
        rows.append(line)
    return boxes, "\n".join(rows) + ("\n" if rows else ""), polygons


def index_sources(root, cache=None):
    # Content-affecting file metadata invalidates the cache, including annotation edits.
    for source in SOURCES:
        if not (root / source).is_dir():
            raise FileNotFoundError(root / source)
    files = sorted(p for source in SOURCES for p in (root / source).glob("*/*/*") if p.is_file())
    fingerprint = hashlib.sha256(
        "\n".join(
            f"{p.relative_to(root)}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in files
        ).encode()
    ).hexdigest()
    if cache and cache.is_file():
        payload = json.loads(cache.read_text())
        if payload.get("fingerprint") == fingerprint and payload.get("version") == 1:
            print("  using validated image feature cache", flush=True)
            return [Record(**r) for r in payload["records"]]
    records = []
    for dataset in SOURCES:
        source = root / dataset
        if not source.is_dir():
            raise FileNotFoundError(source)
        for split in ("train", "valid", "test"):
            for path in sorted((source / split / "images").iterdir()):
                if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    continue
                label = source / split / "labels" / (path.stem + ".txt")
                if not label.is_file():
                    raise FileNotFoundError(f"Missing YOLO annotation: {label}")
                boxes, _, polygons = normalized_labels(label)
                stem = original_stem(path.name)
                kind = (
                    "personal_webcam"
                    if dataset == SOURCES[1]
                    else ("video_frame" if video_key(stem) else "web_photo")
                )
                w, h, ph, dh, color, sharp = image_features(path)
                records.append(
                    Record(
                        str(path.relative_to(root)),
                        str(label.relative_to(root)),
                        dataset,
                        split,
                        stem,
                        kind,
                        w,
                        h,
                        boxes,
                        ph,
                        dh,
                        color,
                        sharp,
                        polygon_boxes=polygons,
                    )
                )
                if len(records) % 200 == 0:
                    print(f"  indexed {len(records)}", flush=True)
    if cache:
        cache.write_text(
            json.dumps(
                {"version": 1, "fingerprint": fingerprint, "records": [asdict(r) for r in records]}
            )
        )
    return records


def canonical_key(r):
    return (r.old_split in ("valid", "test"), r.color > 2, r.sharpness, r.width * r.height, r.path)


def pair_distances(records, p_threshold, d_threshold):
    """Use both hashes from the SAME view pair, including mirrored/cropped views."""
    n = len(records)
    ph = np.asarray([r.phash for r in records], dtype=np.uint64)
    dh = np.asarray([r.dhash for r in records], dtype=np.uint64)
    score = np.full((n, n), np.inf, dtype=np.float32)
    pd = np.zeros((n, n), np.uint8)
    dd = pd.copy()
    if not n:
        return score, pd, dd
    personal = np.asarray([r.source_type == "personal_webcam" for r in records])
    both_personal = personal[:, None] & personal[None, :]
    p_limit = np.where(both_personal, min(p_threshold, 6), p_threshold)
    d_limit = np.where(both_personal, min(d_threshold, 8), d_threshold)
    for a in range(ph.shape[1]):
        for b in range(ph.shape[1]):
            p = np.bitwise_count(ph[:, a, None] ^ ph[None, :, b])
            d = np.bitwise_count(dh[:, a, None] ^ dh[None, :, b])
            s = np.maximum(p / p_limit, d / d_limit)
            better = s < score
            score[better], pd[better], dd[better] = s[better], p[better], d[better]
    return score, pd, dd


def source_pair(a, b):
    return tuple(sorted(((a.dataset, a.original), (b.dataset, b.original))))


def leakage_check(records, p_threshold=16, d_threshold=22, nonduplicate_sources=()):
    """Assert no group or accepted near-duplicate spans splits.

    Hash candidates explicitly rejected by geometric verification are distinct
    photos, not duplicate edges. Their source pairs must be supplied from audit.
    """
    groups = defaultdict(set)
    for r in records:
        groups[r.group].add(r.split)
    spanning = sum(len(s) > 1 for s in groups.values())
    scores, _, _ = pair_distances(records, p_threshold, d_threshold)
    splits = np.asarray([r.split for r in records])
    excluded = set(nonduplicate_sources)
    count = sum(
        source_pair(records[i], records[j]) not in excluded
        for i, j in zip(*np.where(np.triu((scores <= 1) & (splits[:, None] != splits), 1)))
    )
    if spanning or count:
        raise AssertionError(f"Leakage: {spanning} spanning groups, {count} cross-split hash pairs")
    return {"spanning_groups": spanning, "cross_split_hash_pairs": count}


def contact_sheet(root, records, pairs, destination, title, pd, dd, geometric_pairs=()):
    sheet = Image.new("RGB", (1200, 70 + len(pairs[:40]) * 145), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill="black")
    metadata = []
    geometric_pairs = set(geometric_pairs)
    for row, (i, j) in enumerate(pairs[:40]):
        y = 60 + row * 145
        method = "geometry" if (i, j) in geometric_pairs else "hash distance"
        for col, idx in enumerate((i, j)):
            r = records[idx]
            with Image.open(root / r.path) as im:
                sheet.paste(ImageOps.contain(im.convert("RGB"), (260, 112)), (col * 600, y))
            draw.text(
                (col * 600 + 265, y),
                f"{idx}: {r.source_type}\np={pd[i, j]} d={dd[i, j]} ({method})\n{r.original[:43]}",
                fill="black",
            )
        metadata.append(
            {
                "left": records[i].path,
                "right": records[j].path,
                "phash_distance": int(pd[i, j]),
                "dhash_distance": int(dd[i, j]),
                "method": method,
            }
        )
    sheet.save(destination)
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def geometric_matches(root, records, hash_scores, audit):
    """Recover crops/renamed copies without accepting hash-similar printer scenes.

    Cross-stem 16/22 hash candidates require SIFT,
    a spatially distributed homography, and photometric agreement after alignment.
    """
    cv2.setNumThreads(1)
    broad = hash_scores
    # Verify the actual closest copies. A preferred valid/test export may be a
    # very different crop, or even a source-stem collision, from the matched copy.
    best_copies = {}
    for i, j in zip(*np.where(np.triu(broad <= 1, 1))):
        a, b = records[i], records[j]
        if (a.dataset, a.original) == (b.dataset, b.original):
            continue
        if a.source_type == b.source_type == "video_frame" and video_key(a.original) == video_key(
            b.original
        ):
            continue
        key = source_pair(a, b)
        pair = (int(i), int(j))
        previous = best_copies.get(key)
        if previous is None or (broad[pair], pair) < (broad[previous], previous):
            best_copies[key] = pair
    candidates = set(best_copies.values())
    fingerprint = hashlib.sha256(
        "\n".join(
            f"{r.path}:{(root / r.path).stat().st_size}:{(root / r.path).stat().st_mtime_ns}"
            for r in records
        ).encode()
    ).hexdigest()
    cache_path = audit / "geometry_cache.json"
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cache = (
        cached.get("pairs", {})
        if cached.get("fingerprint") == fingerprint and cached.get("version") in (2, 3, 4)
        else {}
    )
    features = {}
    sift = cv2.SIFT_create(nfeatures=1000)

    def feature(idx, flip=False, square=False):
        key = (idx, flip, square)
        if key not in features:
            with Image.open(root / records[idx].path) as im:
                im = im.convert("L")
                im = (
                    im.resize((512, 512), Image.Resampling.LANCZOS)
                    if square
                    else ImageOps.contain(im, (640, 640))
                )
                gray = np.asarray(im)
            if flip:
                gray = np.fliplr(gray).copy()
            equal = cv2.equalizeHist(gray)
            kp, descriptors = sift.detectAndCompute(equal, None)
            features[key] = (gray, kp, descriptors)
        return features[key]

    def verify(i, j, square_only=False):
        best = {"accepted": False, "inliers": 0}
        for flip in (False, True):
            a, ka, da = feature(i, flip, square_only)
            b, kb, db = feature(j, square=square_only)
            if da is None or db is None or len(da) < 12 or len(db) < 12:
                continue
            cv2.setRNGSeed(42)
            matches = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 32}).knnMatch(
                da, db, k=2
            )
            good = [
                m
                for pair in matches
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < 0.72 * n.distance
            ]
            if len(good) < 12:
                continue
            x = np.float32([ka[m.queryIdx].pt for m in good])
            y = np.float32([kb[m.trainIdx].pt for m in good])
            cv2.setRNGSeed(42)
            h, mask = cv2.findHomography(x, y, cv2.RANSAC, 3.0)
            if h is None:
                continue
            inlier = mask.ravel().astype(bool)
            n = int(inlier.sum())
            if n < 12 or n / len(good) < 0.65:
                continue
            coverage = min(
                cv2.contourArea(cv2.convexHull(x[inlier])) / a.size,
                cv2.contourArea(cv2.convexHull(y[inlier])) / b.size,
            )
            if coverage < 0.03:
                continue
            warped = cv2.warpPerspective(a, h, (b.shape[1], b.shape[0]))
            valid = (
                cv2.warpPerspective(
                    np.ones_like(a), h, (b.shape[1], b.shape[0]), flags=cv2.INTER_NEAREST
                )
                > 0
            )
            valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            if valid.sum() < 0.15 * min(a.size, b.size):
                continue
            av, bv = warped[valid].astype(float), b[valid].astype(float)
            corr = float(np.corrcoef(av, bv)[0, 1]) if av.std() > 1 and bv.std() > 1 else 0.0
            ga = cv2.Laplacian(warped, cv2.CV_32F)[valid]
            gb = cv2.Laplacian(b, cv2.CV_32F)[valid]
            edge_corr = float(np.corrcoef(ga, gb)[0, 1]) if ga.std() > 1 and gb.std() > 1 else 0.0
            personal = records[i].source_type == records[j].source_type == "personal_webcam"
            # A static webcam background alone must not identify a frame.
            accepted = bool(
                corr >= (0.985 if personal else 0.85) and edge_corr >= (0.65 if personal else 0.15)
            )
            result = {
                "accepted": accepted,
                "inliers": n,
                "coverage": float(coverage),
                "correlation": corr,
                "edge_correlation": edge_corr,
                "mirrored": flip,
                "square_normalized": square_only,
            }
            if n > best["inliers"] or accepted:
                best = result
            if accepted:
                break
        return best

    accepted, rows = [], []
    print(f"Geometrically verifying {len(candidates)} wider hash candidates...", flush=True)
    for count, (i, j) in enumerate(sorted(candidates)):
        key = records[i].path + "|" + records[j].path
        result = cache.get(key)
        needs_square = result is None or cached.get("version") != 4
        if result is not None and cached.get("version") == 2:
            personal = records[i].source_type == records[j].source_type == "personal_webcam"
            if not personal and "correlation" in result:
                result["accepted"] = bool(
                    result["correlation"] >= 0.85 and result["edge_correlation"] >= 0.15
                )
        if result is None:
            result = verify(i, j)
        if (
            needs_square
            and not result["accepted"]
            and broad[i, j] <= 0.75
            and records[i].source_type == records[j].source_type == "web_photo"
        ):
            # Some exports stretch a whole photo to a radically different aspect ratio.
            alternative = verify(i, j, square_only=True)
            if alternative["accepted"] or alternative["inliers"] > result["inliers"]:
                result = alternative
        cache[key] = result
        rows.append(
            dict(
                left=records[i].path,
                right=records[j].path,
                source_pair=source_pair(records[i], records[j]),
                **result,
            )
        )
        if result["accepted"]:
            accepted.append((i, j))
        if (count + 1) % 200 == 0:
            print(f"  verified {count + 1}; accepted {len(accepted)}", flush=True)
            cache_path.write_text(
                json.dumps({"version": 4, "fingerprint": fingerprint, "pairs": cache})
            )
    cache_path.write_text(
        json.dumps({"version": 4, "fingerprint": fingerprint, "pairs": cache}, indent=2) + "\n"
    )
    (audit / "geometric_matches.json").write_text(json.dumps(rows, indent=2) + "\n")
    accepted_set = set(accepted)
    rejected = {
        source_pair(records[i], records[j]) for i, j in candidates if (i, j) not in accepted_set
    }
    return accepted, len(candidates), rejected


def assign_splits(records, seed):
    groups = defaultdict(list)
    for r in records:
        if r.kept:
            groups[r.group].append(r)
    rng = random.Random(seed)
    for source in ("web_photo", "video_frame", "personal_webcam"):
        members = [g for g in groups.values() if g[0].source_type == source]
        rng.shuffle(members)
        members.sort(key=len, reverse=True)
        ratios = np.array([0.4, 0.1, 0.5] if source == "personal_webcam" else [0.7, 0.1, 0.2])
        vectors = []
        for g in members:
            balanced = g
            if source == "personal_webcam":
                canonical = {}
                for r in g:
                    if r.original not in canonical or canonical_key(r) > canonical_key(
                        canonical[r.original]
                    ):
                        canonical[r.original] = r
                balanced = list(canonical.values())
            vectors.append(
                np.array(
                    [
                        1,
                        len(balanced),
                        sum(1 in r.boxes for r in balanced),
                        sum(r.boxes.count(0) for r in balanced),
                        sum(r.boxes.count(1) for r in balanced),
                    ],
                    float,
                )
            )
        if not vectors:
            continue
        target = ratios[:, None] * np.sum(vectors, axis=0)
        current = np.zeros_like(target)
        for g, v in zip(members, vectors):
            costs = []
            for k in range(3):
                trial = current.copy()
                trial[k] += v
                costs.append(float(np.sum((trial - target) ** 2 / np.maximum(target, 1))))
            chosen = int(np.argmin(costs))
            current[chosen] += v
            for r in g:
                r.split = SPLITS[chosen]
    assigned = {r.group: r.split for r in records if r.kept}
    for r in records:
        r.split = assigned.get(r.group, "")


def build(root, output, seed=42, keep_aug_copies=False, p_threshold=16, d_threshold=22):
    root, output = Path(root).resolve(), Path(output).resolve()
    if (
        output == root
        or output in root.parents
        or any(output == root / s or root / s in output.parents for s in SOURCES)
    ):
        raise ValueError("Output must not overwrite an input dataset")
    output.mkdir(parents=True, exist_ok=True)
    audit = output / "_audit"
    audit.mkdir(exist_ok=True)
    print("Indexing full/cropped/mirrored image hashes...", flush=True)
    records = index_sources(root, audit / "index.json")
    uf = UnionFind(len(records))
    stems, videos = defaultdict(list), defaultdict(list)
    for i, r in enumerate(records):
        stems[(r.dataset, r.original)].append(i)
        if video_key(r.original):
            videos[(r.dataset, video_key(r.original))].append(i)
    for members in list(stems.values()) + list(videos.values()):
        for i in members[1:]:
            uf.union(members[0], i)
    print(f"Comparing {len(records)} images...", flush=True)
    scores, pd, dd = pair_distances(records, p_threshold, d_threshold)
    rows, cols = np.where(np.triu(scores <= 1, 1))
    hash_pairs = list(zip(rows.tolist(), cols.tolist()))
    hash_cross_before = sum(records[i].old_split != records[j].old_split for i, j in hash_pairs)
    geometric_pairs, geometric_candidates, rejected = geometric_matches(
        root, records, scores, audit
    )
    accepted_cross_before = sum(
        records[i].old_split != records[j].old_split
        and source_pair(records[i], records[j]) not in rejected
        for i, j in hash_pairs
    )
    for i, j in geometric_pairs:
        uf.union(i, j)
    components = defaultdict(list)
    for i in range(len(records)):
        components[uf.find(i)].append(i)
    for members in components.values():
        group = (
            "g_"
            + hashlib.sha256(
                "\n".join(sorted(records[i].path for i in members)).encode()
            ).hexdigest()[:16]
        )
        for i in members:
            records[i].group = group
    for members in stems.values():
        if records[members[0]].source_type != "personal_webcam" or not keep_aug_copies:
            best = max(members, key=lambda i: canonical_key(records[i]))
            for i in members:
                if i != best:
                    records[i].kept = False
                    records[i].removed_reason = "same_original_stem"
    # A video is a leakage group, not a single still photo; preserve its source frames.
    for members in components.values():
        web = [i for i in members if records[i].source_type == "web_photo" and records[i].kept]
        if web:
            best = max(web, key=lambda i: canonical_key(records[i]))
            for i in web:
                if i != best:
                    records[i].kept = False
                    records[i].removed_reason = "perceptual_duplicate"
    cross_candidates = [
        (int(i), int(j))
        for i, j in zip(*np.triu_indices(len(records), 1))
        if records[i].group != records[j].group
    ]
    cross_candidates.sort(key=lambda pair: (scores[pair], pair))
    within = sorted(
        [pair for pair in hash_pairs if records[pair[0]].group == records[pair[1]].group]
        + geometric_pairs,
        key=lambda pair: (
            records[pair[0]].original == records[pair[1]].original,
            -scores[pair],
            pair,
        ),
    )

    def distinct(pairs):
        used, result = set(), []
        for i, j in pairs:
            key = tuple(
                sorted(
                    (
                        (records[i].dataset, records[i].original),
                        (records[j].dataset, records[j].original),
                    )
                )
            )
            if key not in used:
                used.add(key)
                result.append((i, j))
            if len(result) == 40:
                break
        return result

    contact_sheet(
        root,
        records,
        distinct(cross_candidates),
        audit / "closest_cross_cluster.jpg",
        "40 closest unmerged source pairs (boundary review)",
        pd,
        dd,
    )
    contact_sheet(
        root,
        records,
        distinct(within),
        audit / "within_threshold.jpg",
        "40 weakest accepted duplicate matches (hash / verified geometry)",
        pd,
        dd,
        geometric_pairs,
    )
    contact_sheet(
        root,
        records,
        distinct(
            [
                pair
                for pair in cross_candidates
                if all(records[i].source_type == "web_photo" for i in pair)
            ]
        ),
        audit / "web_cross_cluster.jpg",
        "40 closest unmerged web-photo pairs",
        pd,
        dd,
    )
    contact_sheet(
        root,
        records,
        distinct(
            [pair for pair in within if all(records[i].source_type == "web_photo" for i in pair)]
        ),
        audit / "web_within_threshold.jpg",
        "40 weakest accepted web-photo pairs",
        pd,
        dd,
        geometric_pairs,
    )
    assign_splits(records, seed)
    kept = [r for r in records if r.kept]
    print("Checking output leakage...", flush=True)
    after = leakage_check(kept, p_threshold, d_threshold, rejected)
    for split in SPLITS:
        folder = output / split
        if folder.exists():
            shutil.rmtree(folder)
        (folder / "images").mkdir(parents=True)
        (folder / "labels").mkdir()
    manifest = []
    for r in kept:
        name = r.dataset + "__" + r.old_split + "__" + Path(r.path).name
        relative = Path(r.split) / "images" / name
        shutil.copy2(root / r.path, output / relative)
        label_dest = output / r.split / "labels" / (Path(name).stem + ".txt")
        if r.polygon_boxes:
            label_dest.write_text(normalized_labels(root / r.label)[1])
        else:
            shutil.copy2(root / r.label, label_dest)
        manifest.append(
            {
                "path": relative.as_posix(),
                "split": r.split,
                "group": r.group,
                "source_type": r.source_type,
                "has_spaghetti": 1 in r.boxes,
                "n_boxes": len(r.boxes),
                "width": r.width,
                "height": r.height,
            }
        )
    (output / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    (output / "data.yaml").write_text(
        f"path: {json.dumps(str(output))}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 2\nnames: [printer-gantry, spaghetti]\n"
    )
    group_json = {}
    for members in components.values():
        g = [records[i] for i in members]
        group_json[g[0].group] = {
            "split": g[0].split,
            "source_type": sorted({r.source_type for r in g}),
            "members": [asdict(r) for r in g],
        }
    (output / "groups.json").write_text(json.dumps(group_json, indent=2) + "\n")
    stem_leak = sum(len({records[i].old_split for i in m}) > 1 for m in stems.values())
    group_leak = sum(len({records[i].old_split for i in m}) > 1 for m in components.values())
    counts = {}
    for split in SPLITS:
        subset = [r for r in kept if r.split == split]
        counts[split] = {
            "images": len(subset),
            "groups": len({r.group for r in subset}),
            "sources": dict(Counter(r.source_type for r in subset)),
            "positive_images": sum(1 in r.boxes for r in subset),
            "printer_gantry_boxes": sum(r.boxes.count(0) for r in subset),
            "spaghetti_boxes": sum(r.boxes.count(1) for r in subset),
        }
    rejected_cross_after = sum(
        records[i].kept
        and records[j].kept
        and records[i].split != records[j].split
        and source_pair(records[i], records[j]) in rejected
        for i, j in hash_pairs
    )
    summary = {
        "rejected_cross_split_candidates": rejected_cross_after,
        "indexed": len(records),
        "kept": len(kept),
        "removed": len(records) - len(kept),
        "groups": len(components),
        "seed": seed,
        "phash_threshold": p_threshold,
        "geometric_candidates": geometric_candidates,
        "geometric_merges": len(geometric_pairs),
        "dhash_threshold": d_threshold,
        "personal_thresholds": {"phash": min(p_threshold, 6), "dhash": min(d_threshold, 8)},
        "removed_by_reason": dict(Counter(r.removed_reason for r in records if not r.kept)),
        "hash_pairs": len(hash_pairs),
        "rejected_source_pairs": len(rejected),
        "before": {
            "cross_split_confirmed_duplicate_pairs": accepted_cross_before,
            "spanning_stems": stem_leak,
            "spanning_stems_by_dataset": {
                source: sum(
                    len({records[i].old_split for i in m}) > 1
                    for (dataset, _), m in stems.items()
                    if dataset == source
                )
                for source in SOURCES
            },
            "spanning_groups": group_leak,
            "cross_split_hash_pairs": hash_cross_before,
        },
        "after": after,
        "splits": counts,
    }
    (audit / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = [
        "# Real v1 dataset",
        "",
        f"Seed: {seed}. Keep augmentation copies: {keep_aug_copies}.",
        f"Indexed {len(records)} images; kept {len(kept)}; removed {len(records) - len(kept)}; {len(components)} groups.",
        "Removed by reason: " + json.dumps(summary["removed_by_reason"], sort_keys=True) + ".",
        "Removed by input dataset: "
        + json.dumps(dict(Counter(r.dataset for r in records if not r.kept)), sort_keys=True)
        + ".",
        f"Hash-candidate rule: pHash <= {p_threshold}/64 AND dHash <= {d_threshold}/64 on the same view pair. Cross-stem candidates require geometric confirmation.",
        "Views: original and horizontal mirror, each at full / central 90% / central 80% crop. Strong rejected web candidates are also verified after square normalization to recover nonuniformly resized exports.",
        "Personal-to-personal hash limits are capped at pHash 6/dHash 8 to avoid conflating distinct toolhead/bed views in a fixed scene.",
        f"Cross-stem candidates (pHash <= {p_threshold}, dHash <= {d_threshold}; personal capped at 6/8) merge only after SIFT/homography verification: >=12 inliers, >=65% inliers, >=3% spatial coverage, aligned intensity correlation >=0.85 and edge correlation >=0.15. Personal pairs require >=0.985 intensity / >=0.65 edge correlation because their static background is shared by distinct frames.",
        f"Geometric verification accepted {len(geometric_pairs)} of {geometric_candidates} source-stem pairs; details: _audit/geometric_matches.json.",
        "Original stems and common MOV/MP4/AVI prefixes merge unconditionally.",
        "Threshold review: web edge correlation was relaxed from 0.35 to 0.15 after inspecting geometrically aligned, visibly identical resized/JPEG copies with edge correlations 0.19-0.34. Hash 6/8 missed visibly identical renamed/cropped web photos at 8/9 and 10/13. A hash-only 12/16 rule also matched unrelated scenes (filament nest versus printer bed). Therefore 16/22 is only a candidate boundary; spatial and photometric verification decide cross-stem merges. Personal thresholds are tighter because the shared printer background creates many false candidates.",
        "",
        "## Leakage",
        "",
        f"Before: {stem_leak} spanning source stems, {group_leak} spanning groups, {accepted_cross_before} confirmed cross-split duplicate pairs ({hash_cross_before} raw cross-split hash candidates).",
        f"After: {after['spanning_groups']} spanning groups, {after['cross_split_hash_pairs']} accepted cross-split duplicate pairs. Geometrically rejected hash candidates are excluded from duplicate edges (see geometric_matches.json).",
        "",
        "## Split counts",
        "",
        "|Split|Images|Groups|Web|Video|Personal|Spaghetti images|Gantry boxes|Spaghetti boxes|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, c in counts.items():
        report.append(
            f"|{split}|{c['images']}|{c['groups']}|{c['sources'].get('web_photo', 0)}|{c['sources'].get('video_frame', 0)}|{c['sources'].get('personal_webcam', 0)}|{c['positive_images']}|{c['printer_gantry_boxes']}|{c['spaghetti_boxes']}|"
        )
    report += [
        "",
        "## Image size histogram (kept)",
        "",
        "```json",
        json.dumps(dict(Counter(f"{r.width}x{r.height}" for r in kept)), indent=2),
        "```",
        "",
        "## Audit and limitations",
        "",
        "See `_audit/closest_cross_cluster.jpg` and `_audit/within_threshold.jpg`; additional web-only sheets isolate web-photo boundaries. JSON sidecars identify every pair.",
        f"After splitting, {rejected_cross_after} cross-split hash candidates were explicitly rejected as different scenes/frames by geometric verification; they are not duplicate edges.",
        "Personal copies: prefer valid/test, then non-grayscale, then sharpness. One per original frame unless requested otherwise.",
        "Web photos retain one canonical per perceptual cluster; video frames retain one per original stem. Discarded member metadata and reasons remain in groups.json.",
        "Source stems are trusted provenance: collisions and substantially different exports are conservatively grouped. Cross-stem geometry is checked on the closest actual copies, rather than only the preferred canonical export.",
        "Targets are 70/10/20 per source, with personal frames 40/10/50; indivisible video/duplicate groups prevent exact proportions.",
        "Stratification balances group counts, image counts, positive-image counts and each class's box counts.",
        f"Image bytes are copied unchanged. {sum(r.polygon_boxes for r in kept)} polygon annotations are converted to enclosing YOLO boxes; other labels are copied. Manifest paths are relative to this dataset.",
        "Hashes are heuristic: severe off-center crops, rotations, overlays and visually similar scenes need manual review. This data cannot establish temporal performance or false-pause rate.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def check_dataset(output):
    """Recheck a built dataset using its recorded thresholds and geometric verdicts."""
    output = Path(output)
    summary = json.loads((output / "_audit" / "summary.json").read_text())
    groups = json.loads((output / "groups.json").read_text())
    for group_id, group in groups.items():
        if any(r["group"] != group_id or r["split"] != group["split"] for r in group["members"]):
            raise AssertionError(f"Inconsistent group metadata: {group_id}")
    records = [Record(**r) for g in groups.values() for r in g["members"] if r["kept"]]
    verdicts = json.loads((output / "_audit" / "geometric_matches.json").read_text())
    rejected = {
        tuple(tuple(x) for x in row["source_pair"]) for row in verdicts if not row["accepted"]
    }
    return leakage_check(records, summary["phash_threshold"], summary["dhash_threshold"], rejected)


def add_arguments(parser):
    sub = parser.add_subparsers(dest="realdata_action", required=True)
    command = sub.add_parser("build", help="Build grouped leakage-free real_v1 YOLO data")
    command.add_argument("--datasets-root", type=Path, default=REAL_DATASETS)
    command.add_argument("--output", type=Path)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--keep-aug-copies", action="store_true")
    command.add_argument("--phash-threshold", type=int, default=16)
    command.add_argument("--dhash-threshold", type=int, default=22)


def run(args):
    if not 1 <= args.phash_threshold <= 64 or not 1 <= args.dhash_threshold <= 64:
        raise ValueError("Hash thresholds must be in [1, 64]")
    build(
        args.datasets_root,
        args.output or args.datasets_root / "real_v1",
        args.seed,
        args.keep_aug_copies,
        args.phash_threshold,
        args.dhash_threshold,
    )
    return 0
