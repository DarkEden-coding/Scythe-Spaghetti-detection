"""Resumable OrcaSlicer 2.4.2 Flatpak slices using flattened system presets."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import tempfile
import time

import trimesh

from .paths import CORPUS_DIR, GCODE_CACHE
from .corpus import atomic_json

PROFILES_DIR = Path(__file__).with_name("profiles")
PROFILES = {"bambu_x1c": (256, 256), "prusa_mk4": (250, 210),
            "voron24_300": (300, 300), "bambu_a1mini": (180, 180)}
PATTERNS = ("grid", "gyroid", "rectilinear", "cubic", "honeycomb", "lightning")
SCHEMA_VERSION = 1
SLICE_IMPLEMENTATION_VERSION = 5


def _profile(name: str, kind: str) -> dict:
    if name not in PROFILES:
        raise ValueError(f"Unknown printer profile: {name}")
    return json.loads((PROFILES_DIR / f"{name}_{kind}.json").read_text())


@lru_cache(maxsize=1)
def slicer_version() -> str:
    version_dir = GCODE_CACHE / ".orca-version"
    version_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["flatpak", "run", "--branch=stable",
                             f"--filesystem={version_dir.resolve()}",
                             "com.orcaslicer.OrcaSlicer", "--help"],
                            cwd=version_dir, capture_output=True, text=True, timeout=30, check=True)
    match = re.search(r"OrcaSlicer[- ]([\w.\-]+)", result.stdout)
    if not match or match[1] != "2.4.2":
        raise RuntimeError(f"Expected stable OrcaSlicer 2.4.2: {result.stdout[:100]}")
    return f"OrcaSlicer {match[1]}"


def normalize_settings(settings: dict) -> dict:
    defaults = {"layer_height": 0.2, "infill_pattern": "grid", "infill_density": 0.15,
                "supports": False, "support_type": "normal", "brim": False,
                "scale": 1.0, "copies": 1, "wall_loops": 2}
    unknown = set(settings) - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown slice settings: {sorted(unknown)}")
    s = defaults | settings
    for key in ("layer_height", "infill_density", "scale"):
        s[key] = float(s[key])
        if not math.isfinite(s[key]):
            raise ValueError(f"Nonfinite {key}")
    if not 0.12 <= s["layer_height"] <= 0.28:
        raise ValueError("layer_height must be 0.12–0.28 mm")
    if not 0.05 <= s["infill_density"] <= 0.40:
        raise ValueError("infill_density must be a fraction 0.05–0.40")
    if not 0.001 <= s["scale"] <= 100:
        raise ValueError("scale must be a uniform multiplier 0.001–100")
    if s["infill_pattern"] not in PATTERNS or s["support_type"] not in ("normal", "tree"):
        raise ValueError("Unsupported infill pattern or support type")
    for key, lo, hi in (("copies", 1, 4), ("wall_loops", 2, 4)):
        if type(s[key]) is not int or not lo <= s[key] <= hi:
            raise ValueError(f"{key} must be an integer {lo}–{hi}")
    for key in ("supports", "brim"):
        if type(s[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    return s


def sample_settings(entry: dict, printer_profile: str, rng: random.Random) -> dict:
    copies = rng.randint(1, 4)
    grid = math.ceil(math.sqrt(copies))
    size = entry["bbox_mm"]
    # Leave 12 mm per occupied grid cell for brim/support and arrangement spacing.
    target = min(PROFILES[printer_profile]) * rng.uniform(0.25, min(0.85, 0.90 / grid))
    scale = min(target / max(size[:2]), 100 / size[2], 100)
    scale = min(scale, (min(PROFILES[printer_profile]) / grid - 12) / max(size[:2]))
    return normalize_settings({"layer_height": rng.choice([0.12, 0.16, 0.20, 0.24, 0.28]),
                               "infill_pattern": rng.choice(PATTERNS),
                               "infill_density": round(rng.uniform(0.05, 0.40), 3),
                               "supports": bool(entry.get("has_overhangs")) and rng.choice([True, False]),
                               "support_type": rng.choice(["normal", "tree"]),
                               "brim": rng.choice([True, False]), "scale": scale,
                               "copies": copies, "wall_loops": rng.randint(2, 4)})


def add_comment_aliases(path: Path) -> None:
    """Preserve Bambu labels and add equivalent Prusa-style parser aliases.

    Orca's Bambu writer always uses LINE_WIDTH/LAYER_HEIGHT/Z_HEIGHT labels,
    regardless of gcode_comments. Only comments are added; moves are untouched.
    """
    text = path.read_text()
    aliases = {"LINE_WIDTH": "WIDTH", "LAYER_HEIGHT": "HEIGHT", "Z_HEIGHT": "Z"}
    pattern = r"^(;[ \t]*(LINE_WIDTH|LAYER_HEIGHT|Z_HEIGHT):[ \t]*([^\r\n]+))$"
    text = re.sub(pattern, lambda m: m[1] + "\n;" + aliases[m[2]] + ":" + m[3],
                  text, flags=re.M)
    path.write_text(text)


def gcode_details(path: Path) -> dict:
    text = path.read_text(errors="replace")
    markers = {"feature": bool(re.search(r"^;\s*(?:TYPE:|FEATURE:)", text, re.M)),
               "width": bool(re.search(r"^;\s*WIDTH:", text, re.M)),
               "height": bool(re.search(r"^;\s*HEIGHT:", text, re.M)),
               "layer": bool(re.search(r"^;\s*(?:CHANGE_LAYER|LAYER_CHANGE|BEFORE_LAYER_CHANGE)", text, re.M))}
    arcs = len(re.findall(r"^\s*G0?[23](?:\s|$)", text, re.M))
    estimate = re.search(r"total estimated time\s*[:=]\s*([^;\n]+)", text)
    if not estimate:
        estimate = re.search(r"(?:estimated printing time \(normal mode\)|model printing time)\s*[:=]\s*([^;\n]+)", text)
    seconds = None
    if estimate:
        units = {"d": 86400, "h": 3600, "m": 60, "s": 1}
        parts = re.findall(r"([\d.]+)\s*([dhms])", estimate[1])
        if parts:
            seconds = sum(float(value) * units[unit] for value, unit in parts)
    return {"comment_markers": markers, "arc_moves": arcs, "estimated_print_time_s": seconds}


def _execute(command: list[str], work: Path, timeout: float) -> None:
    # Orca launches native worker threads. Kill the whole process group on timeout.
    with (work / "slicer.log").open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                cwd=work, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise
    if code:
        raise RuntimeError(f"Orca exited {code}; see slicer.log and orca.log in the slice directory")


def _record_failure(record: dict) -> None:
    path = GCODE_CACHE / "failed.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record) + "\n")
        f.flush()


def slice_model(model_entry: dict, printer_profile: str, settings: dict) -> Path:
    """Return model.gcode, reusing only a successful identical cached slice.

    ``scale`` is an absolute multiplier of the original STL. Caller samples it;
    oversized input is rejected, never silently resized. Timeout is controlled
    by SCYTHE_SLICE_TIMEOUT (seconds, default 180). Failure records are retained.
    """
    s = normalize_settings(settings)
    machine = _profile(printer_profile, "machine")
    process = _profile(printer_profile, "process")
    filament = _profile(printer_profile, "filament")
    model_id = model_entry["id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", model_id):
        raise ValueError("Unsafe model id")
    stl = Path(model_entry["stl"]).resolve(strict=True)
    size = model_entry.get("bbox_mm")
    if size and (max(size[:2]) * s["scale"] > min(PROFILES[printer_profile]) - 10
                 or size[2] * s["scale"] > 100.001):
        raise ValueError("Scaled model exceeds plate margin or 100 mm height")
    version = slicer_version()
    identity = {"schema_version": SCHEMA_VERSION, "implementation_version": SLICE_IMPLEMENTATION_VERSION,
                "printer_profile": printer_profile,
                "settings": s, "slicer": version, "model_id": model_id,
                "model_license": model_entry["license"],
                "stl_sha256": hashlib.sha256(stl.read_bytes()).hexdigest(),
                "profiles": [machine, process, filament]}
    slice_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    parent = GCODE_CACHE / printer_profile / model_id
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / slice_hash
    with (parent / f".{slice_hash}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata_path = target / "slice.json"
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text())
            if (previous.get("success")
                    and previous.get("orca_result", {}).get("return_code") == 0
                    and (target / "model.gcode").is_file()):
                return target / "model.gcode"
        work = Path(tempfile.mkdtemp(prefix=f".{slice_hash}.", dir=parent))
        # Explicit overrides keep slicer-generated toolpath annotations enabled.
        process.update({"layer_height": str(s["layer_height"]),
                        "sparse_infill_pattern": s["infill_pattern"],
                        "sparse_infill_density": f"{s['infill_density'] * 100:g}%",
                        "enable_support": "1" if s["supports"] else "0",
                        "support_type": s["support_type"] + "(auto)",
                        "brim_type": "outer_only" if s["brim"] else "no_brim",
                        "brim_width": "5" if s["brim"] else "0",
                        "wall_loops": str(s["wall_loops"]), "enable_arc_fitting": "0",
                        "gcode_comments": "1", "exclude_object": "1",
                        "enable_prime_tower": "0", "curr_bed_type": "Textured PEI Plate",
                        "filename_format": "model.gcode"})
        machine.update({"gcode_comments": "1"})
        for kind, data in (("machine", machine), ("process", process), ("filament", filament)):
            atomic_json(work / f"{kind}.json", data)
        # Keep Flatpak app settings and temporary files off /home as well.
        datadir = work / "orca-data"
        datadir.mkdir()
        command = ["flatpak", "run", "--branch=stable", "--filesystem=/mnt/Primary",
                   f"--filesystem={stl.parent}", f"--filesystem={PROFILES_DIR}",
                   f"--filesystem={work}", f"--env=TMPDIR={work}",
                   "--env=OMP_NUM_THREADS=2", "com.orcaslicer.OrcaSlicer",
                   "--datadir", str(datadir), "--debug", "3", "--logfile", str(work / "orca.log"),
                   "--load-settings",
                   f"{work / 'machine.json'};{work / 'process.json'}",
                   "--load-filaments", str(work / "filament.json"),
                   "--clone-objects", str(s["copies"]),
                   "--arrange", "1", "--orient", "0", "--ensure-on-bed",
                   "--slice", "1", "--outputdir", str(work), str(work / "input.stl")]
        record = {"schema_version": SCHEMA_VERSION, "slice_hash": slice_hash,
                  "model_id": model_id, "model": dict(model_entry),
                  "printer_profile": printer_profile, "settings": s,
                  "slicer": version, "slicer_version": "2.4.2", "success": False,
                  "gcode": str(target / "model.gcode"), "command": command,
                  "bed_size_mm": list(PROFILES[printer_profile]),
                  "plate_center_mm": [x / 2 for x in PROFILES[printer_profile]],
                  "gcode_flavor": machine["gcode_flavor"],
                  "nozzle_diameter": 0.4, "filament_diameter": 1.75,
                  "machine_acceleration_mm_s2": float(machine.get("machine_max_acceleration_extruding", [5000])[0]),
                  "stl_sha256": identity["stl_sha256"]}
        start = time.monotonic()
        error = None
        try:
            # Orca 2.4.2 crashes when CLI --scale enlarges raw STLs (>1x).
            # Apply the exact caller multiplier before loading/arranging instead.
            scaled_mesh = trimesh.load_mesh(stl, process=False)
            scaled_mesh.apply_scale(s["scale"])
            scaled_mesh.export(work / "input.stl")
            _execute(command, work, float(os.environ.get("SCYTHE_SLICE_TIMEOUT", "180")))
            record["orca_result"] = json.loads((work / "result.json").read_text())
            if record["orca_result"].get("return_code") != 0:
                raise RuntimeError(f"Orca result: {record['orca_result']}")
            gcodes = sorted(work.rglob("*.gcode"))
            if len(gcodes) != 1:
                raise RuntimeError(f"Expected one plate of G-code, got {len(gcodes)}; copies may not fit")
            produced = next((p for p in gcodes if p.name == "model.gcode"), gcodes[0])
            if produced != work / "model.gcode":
                shutil.move(produced, work / "model.gcode")
            add_comment_aliases(work / "model.gcode")
            record.update(gcode_details(work / "model.gcode"))
            if not all(record["comment_markers"].values()):
                raise RuntimeError(f"Missing timeline comments: {record['comment_markers']}")
            record["success"] = True
        except Exception as exc:
            error = exc
            if (work / "result.json").is_file():
                try:
                    record["orca_result"] = json.loads((work / "result.json").read_text())
                except (ValueError, OSError):
                    pass
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["slice_seconds"] = time.monotonic() - start
        shutil.rmtree(datadir, ignore_errors=True)
        atomic_json(work / "slice.json", record)
        if target.exists():
            shutil.rmtree(target)
        work.replace(target)
        if error:
            _record_failure(record)
            raise RuntimeError(f"Slice failed: {metadata_path}: {error}") from error
        return target / "model.gcode"


def pool(printer_profile: str) -> list[dict]:
    """Read successful slice.json records, with absolute G-code paths."""
    if printer_profile not in PROFILES:
        raise ValueError(f"Unknown printer profile: {printer_profile}")
    records = []
    for path in sorted((GCODE_CACHE / printer_profile).glob("*/*/slice.json")):
        try:
            data = json.loads(path.read_text())
            gcode = path.with_name("model.gcode")
            if data.get("success") and gcode.is_file():
                data["gcode"] = str(gcode.resolve())
                records.append(data)
        except (ValueError, OSError):
            continue
    return records


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profiles", nargs="+", default=["all"])
    parser.add_argument("--per-profile", type=int, default=150)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180)


def run(args) -> int:
    profiles = list(PROFILES) if args.profiles == ["all"] else args.profiles
    if any(p not in PROFILES for p in profiles):
        raise ValueError(f"Profiles must be all or one of {list(PROFILES)}")
    if args.per_profile < 1 or args.workers < 1 or args.timeout <= 0:
        raise ValueError("per-profile, workers, and timeout must be positive")
    os.environ["SCYTHE_SLICE_TIMEOUT"] = str(args.timeout)
    entries = json.loads((CORPUS_DIR / "index.json").read_text())
    if not entries:
        raise ValueError("Empty corpus; run synthpipe corpus first")
    slicer_version()
    for profile in profiles:
        remaining = max(0, args.per_profile - len(pool(profile)))
        rng = random.Random(f"{args.seed}:{profile}")
        shuffled = list(entries)
        rng.shuffle(shuffled)
        # Bounded retries, submitting only the number of missing successes.
        attempts = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            while remaining and attempts < args.per_profile * 5:
                count = min(remaining, args.per_profile * 5 - attempts)
                futures = []
                for _ in range(count):
                    entry = shuffled[attempts % len(shuffled)]
                    attempts += 1
                    try:
                        settings = sample_settings(entry, profile, rng)
                    except (ValueError, KeyError) as exc:
                        _record_failure({"success": False, "model_id": entry.get("id"),
                                         "printer_profile": profile, "error": str(exc)})
                        print(f"Skipping {entry.get('id')}: {exc}", flush=True)
                        continue
                    futures.append(executor.submit(slice_model, entry, profile, settings))
                for future in as_completed(futures):
                    try:
                        print(f"Sliced {future.result()}", flush=True)
                    except Exception as exc:
                        print(str(exc), flush=True)
                remaining = max(0, args.per_profile - len(pool(profile)))
        print(f"{profile}: {len(pool(profile))} successful slices; {remaining} missing", flush=True)
    return int(any(len(pool(p)) < args.per_profile for p in profiles))
