"""Provenance, transformed duplicates, leakage rejection, and an end-to-end build."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from synthpipe.realdata import (
    SOURCES,
    Record,
    assign_splits,
    build,
    image_features,
    leakage_check,
    original_stem,
    pair_distances,
    video_key,
)


def record(name, ph=0, dh=0, **kwargs):
    return Record(
        name,
        name + ".txt",
        SOURCES[0],
        "train",
        name,
        "web_photo",
        100,
        100,
        [1],
        [ph],
        [dh],
        20.0,
        100.0,
        **kwargs,
    )


def test_names_and_video_provenance():
    assert original_stem("photo_with_underlines_png.rf.abcdef.jpg") == "photo_with_underlines"
    assert original_stem("0_4_png.rf.abc.jpg") == "0_4"
    assert video_key("phone_MOV-17") == video_key("phone_MOV-18") == "phone_MOV"
    assert video_key("a_photo") is None


def test_joint_hashes_use_same_view_pair():
    a = replace(record("a"), phash=[0, 0], dhash=[0, 0])
    # Each hash separately matches a view, but neither joint view passes.
    b = replace(record("b"), phash=[0, (1 << 64) - 1], dhash=[(1 << 64) - 1, 0])
    score, _, _ = pair_distances([a, b], 6, 8)
    assert score[0, 1] > 1


def test_leakage_rejects_groups_and_independent_hash_matches():
    a = record("a", group="g1", split="train")
    b = record("b", group="g2", split="test")
    with pytest.raises(AssertionError, match="cross-split hash pairs"):
        leakage_check([a, b])
    b.phash = b.dhash = [(1 << 64) - 1]
    b.group = "g1"
    with pytest.raises(AssertionError, match="spanning groups"):
        leakage_check([a, b])
    b.group = "g2"
    assert leakage_check([a, b]) == {"spanning_groups": 0, "cross_split_hash_pairs": 0}


def test_flip_resize_crop_hashes(tmp_path):
    rng = np.random.default_rng(5)
    image = Image.fromarray(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8))
    # Smooth texture retains structure under resizing and compression.
    image = image.resize((40, 30)).resize((400, 300))
    paths = [tmp_path / f"{i}.png" for i in range(3)]
    image.save(paths[0])
    ImageOps.mirror(image).resize((200, 150)).save(paths[1])
    image.crop((20, 15, 380, 285)).save(paths[2])
    records = []
    for path in paths:
        w, h, ph, dh, c, sharp = image_features(path)
        records.append(Record(str(path), "", "", "", "", "web_photo", w, h, [1], ph, dh, c, sharp))
    scores, _, _ = pair_distances(records, 6, 8)
    assert scores[0, 1] <= 1
    assert scores[0, 2] <= 1


def test_assignment_is_deterministic_and_keeps_groups():
    records = [record(str(i), group=f"g{i // 2}") for i in range(100)]
    assign_splits(records, 42)
    first = [r.split for r in records]
    assign_splits(records, 42)
    assert first == [r.split for r in records]
    assert all(records[i].split == records[i + 1].split for i in range(0, 100, 2))
    assert set(first) == {"train", "val", "test"}


def test_build_copies_bytes_and_removes_personal_augments(tmp_path):
    root = tmp_path / "inputs"
    rng = np.random.default_rng(12)
    for dataset in SOURCES:
        for split in ("train", "valid", "test"):
            for part in ("images", "labels"):
                (root / dataset / split / part).mkdir(parents=True)
        for i in range(8):
            image = Image.fromarray(rng.integers(0, 256, (100, 120, 3), dtype=np.uint8))
            image = image.resize((20, 16)).resize((120, 100))
            copies = ("train", "valid") if dataset == SOURCES[1] else ("train",)
            for split in copies:
                name = f"frame{i}_png.rf.{split}.jpg"
                folder = root / dataset / split
                image.save(folder / "images" / name)
                label = "0 0.5 0.5 0.5 0.5\n" if dataset == SOURCES[1] else "1 0.5 0.5 0.5 0.5\n"
                (folder / "labels" / Path(name).with_suffix(".txt")).write_text(label)
    output = tmp_path / "out"
    summary = build(root, output)
    assert summary["indexed"] == 24
    assert summary["removed"] == 8
    assert summary["after"] == {"spanning_groups": 0, "cross_split_hash_pairs": 0}
    groups = json.loads((output / "groups.json").read_text())
    members = [r for g in groups.values() for r in g["members"]]
    assert sum(r["kept"] and r["source_type"] == "personal_webcam" for r in members) == 8
    assert all(
        r["old_split"] == "valid"
        for r in members
        if r["kept"] and r["source_type"] == "personal_webcam"
    )
    manifest = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert len(manifest) == 16
    original_bytes = {
        r["dataset"] + "__" + r["old_split"] + "__" + Path(r["path"]).name: (
            root / r["path"]
        ).read_bytes()
        for r in members
        if r["kept"]
    }
    for row in manifest:
        assert (output / row["path"]).read_bytes() == original_bytes[Path(row["path"]).name]
        assert (
            output / row["split"] / "labels" / Path(row["path"]).with_suffix(".txt").name
        ).is_file()
    # Regenerating does not accumulate stale output files.
    (output / "train" / "images" / "stale.jpg").write_bytes(b"stale")
    build(root, output)
    assert not (output / "train" / "images" / "stale.jpg").exists()
    assert manifest == [
        json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()
    ]

    augmented = build(root, output, keep_aug_copies=True)
    assert augmented["kept"] == 24


def test_polygon_annotations_become_enclosing_boxes(tmp_path):
    from synthpipe.realdata import normalized_labels

    label = tmp_path / "polygon.txt"
    label.write_text("1 0.1 0.2 0.7 0.2 0.8 0.9 0.1 0.8\n")
    classes, normalized, n_polygons = normalized_labels(label)
    assert classes == [1] and n_polygons == 1
    assert [float(x) for x in normalized.split()[1:]] == pytest.approx([0.45, 0.55, 0.7, 0.7])


def test_real_built_dataset_has_no_leakage():
    from synthpipe.paths import REAL_DATASETS

    output = REAL_DATASETS / "real_v1"
    if not (output / "groups.json").exists():
        pytest.skip("Run synthpipe realdata build to check the downloaded real dataset")
    summary = json.loads((output / "_audit" / "summary.json").read_text())
    groups = json.loads((output / "groups.json").read_text())
    records = [Record(**r) for g in groups.values() for r in g["members"] if r["kept"]]
    verdicts = json.loads((output / "_audit" / "geometric_matches.json").read_text())
    rejected = {tuple(tuple(x) for x in r["source_pair"]) for r in verdicts if not r["accepted"]}
    assert leakage_check(
        records, summary["phash_threshold"], summary["dhash_threshold"], rejected
    ) == {
        "spanning_groups": 0,
        "cross_split_hash_pairs": 0,
    }


def test_geometric_verification_recovers_flipped_crop_and_rejects_other_photo(tmp_path):
    import cv2

    from synthpipe.realdata import geometric_matches

    rng = np.random.default_rng(41)
    texture = cv2.GaussianBlur(rng.integers(0, 256, (500, 500), dtype=np.uint8), (3, 3), 0.7)
    base = Image.fromarray(texture)
    base.save(tmp_path / "a.png")
    ImageOps.mirror(base.crop((50, 70, 450, 470))).save(tmp_path / "b.png")
    Image.fromarray(rng.integers(0, 256, (500, 500), dtype=np.uint8)).save(tmp_path / "c.png")
    records = [record(name) for name in ("a.png", "b.png", "c.png")]
    audit = tmp_path / "audit"
    audit.mkdir()
    accepted, candidates, _rejected = geometric_matches(tmp_path, records, np.zeros((3, 3)), audit)
    assert candidates == 3
    assert accepted == [(0, 1)]


def test_geometry_uses_closest_copy_instead_of_unrelated_preferred_export(tmp_path):
    from synthpipe.realdata import geometric_matches

    rng = np.random.default_rng(16)
    original = Image.fromarray(rng.integers(0, 256, (400, 400), dtype=np.uint8))
    different = Image.fromarray(rng.integers(0, 256, (400, 400), dtype=np.uint8))
    different.save(tmp_path / "preferred.png")
    original.save(tmp_path / "train.png")
    original.save(tmp_path / "other.png")
    records = [record(name) for name in ("preferred.png", "train.png", "other.png")]
    records[0].original = records[1].original = "source_a"
    records[0].old_split = "valid"
    records[2].original = "source_b"
    scores = np.array([[0.0, 0.5, 0.8], [0.5, 0.0, 0.0], [0.8, 0.0, 0.0]])
    audit = tmp_path / "audit"
    audit.mkdir()
    accepted, candidates, _rejected = geometric_matches(tmp_path, records, scores, audit)
    assert candidates == 1
    assert accepted == [(1, 2)]


def test_geometry_recovers_nonuniformly_resized_photo(tmp_path):
    from synthpipe.realdata import geometric_matches

    rng = np.random.default_rng(10)
    image = Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8)).resize((500, 500))
    image.save(tmp_path / "wide.png")
    image.resize((250, 500)).save(tmp_path / "narrow.png")
    records = [record("wide.png"), record("narrow.png")]
    audit = tmp_path / "audit"
    audit.mkdir()
    accepted, _, _rejected = geometric_matches(tmp_path, records, np.zeros((2, 2)), audit)
    assert accepted == [(0, 1)]
