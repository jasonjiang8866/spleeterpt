"""Copy only pinned validation inputs; object storage never holds live training state."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path, PurePosixPath

from global_storage import SOURCE_ID, valid_id
from quality import ROOT, digest

PREFIX = "step90-validation-v1"
TRAIN_PREFIX = "musdb18-training-v1"
MAX_BYTES = 8 * 1024**3
EXPERIMENT = "41189465452-1"
BACKUP = "experiments/" + EXPERIMENT + "/backups/"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def evidence():
    return json.loads((ROOT / "docs/results/persistent-pilot-41189465452.json").read_text())


def generated(provenance_json, *, training=False):
    if training:
        pinned = json.loads((ROOT / "docs/results/musdb18-training-provenance.json").read_text())
        provenance = json.loads(provenance_json)
        files = {"musdb18/provenance.json": provenance_json.encode()}
        for split, tracks in pinned["split_track_ids"].items():
            raw = "".join(
                json.dumps({"track": "/data/musdb18/train-pool/" + t.removeprefix("train/")}) + "\n"
                for t in tracks
            ).encode()
            if hashlib.sha256(raw).hexdigest() != provenance["manifests"][split + ".jsonl"]:
                raise ValueError("Training manifest reconstruction differs from pinned bytes")
            files["musdb18/manifests/" + split + ".jsonl"] = raw
        return files
    backup = evidence()["resume"]["backup"]
    return {
        "musdb18/provenance.json": provenance_json.encode(),
        BACKUP + "latest.json": encoded({"generation": backup["generation"]}),
        BACKUP + backup["generation"] + "/manifest.json": encoded(backup),
    }


def manifest(provenance_json, *, training=False):
    record = evidence()
    pinned = json.loads((ROOT / "docs/results/musdb18-training-provenance.json").read_text())
    if (
        hashlib.sha256(provenance_json.encode()).hexdigest()
        != record["resume"]["dataset"]["provenance_sha256"]
    ):
        raise ValueError("Source provenance checksum mismatch")
    provenance = json.loads(provenance_json)
    if (
        len(provenance["tracks"]) != 100
        or provenance["archive_sha256"] != pinned["archive_sha256"]
        or provenance["test_audio_extracted"] is not False
        or {t["id"]: (t["source_sha256"], t["samples"]) for t in provenance["tracks"]}
        != {t["id"]: (t["source_sha256"], t["samples"]) for t in pinned["tracks"]}
    ):
        raise ValueError("Source inventory mismatch")
    selected = set(pinned["split_track_ids"]["validation"])
    if training:
        selected.update(pinned["split_track_ids"]["train"])
    files = {}
    found = set()
    for track in provenance["tracks"]:
        track_id = track["id"].removesuffix(".stem.mp4")
        if track_id not in selected:
            continue
        found.add(track_id)
        if set(track["files"]) != {"mixture", "vocals", "drums", "bass", "other"}:
            raise ValueError("Stem inventory mismatch")
        for stem, info in track["files"].items():
            files[f"musdb18/train-pool/{PurePosixPath(track_id).name}/{stem}.wav"] = {
                k: info[k] for k in ("bytes", "sha256")
            }
    count = 100 if training else 10
    if found != selected or len(selected) != count or len(files) != count * 5:
        raise ValueError("Expected complete pinned songs")
    if not training:
        backup = record["resume"]["backup"]
        for name, spec in backup["files"].items():
            files[BACKUP + backup["generation"] + "/" + name] = spec
    for name, data in generated(provenance_json, training=training).items():
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    total = sum(v["bytes"] for v in files.values())
    if not 0 < total <= (40 * 1024**3 if training else MAX_BYTES):
        raise ValueError(f"Global snapshot exceeds {40 if training else 8} GiB bound")
    return {
        "format": 1,
        "kind": TRAIN_PREFIX if training else PREFIX,
        "source_volume_id": SOURCE_ID,
        "experiment": EXPERIMENT,
        "selected_tracks": sorted(selected),
        "verified_audio_files": count * 5,
        "bytes": total,
        "files": files,
    }


def checked_path(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError("Unsafe snapshot path")
    target = root / name
    if any(p.is_symlink() for p in [target, *target.parents] if p != root.parent):
        raise ValueError("Snapshot symlink rejected")
    return target


def copy_file(source, target, spec, data=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if (
        target.exists()
        and target.stat().st_size == spec["bytes"]
        and digest(target) == spec["sha256"]
    ):
        return
    h, count = hashlib.sha256(), 0
    with target.open("wb") as dst:
        if data is not None:
            dst.write(data)
            h.update(data)
            count = len(data)
        else:
            with source.open("rb") as src:
                while chunk := src.read(4 * 1024**2):
                    count += len(chunk)
                    if count > spec["bytes"]:
                        raise ValueError("Source file exceeds pinned size")
                    h.update(chunk)
                    dst.write(chunk)
    if count != spec["bytes"] or h.hexdigest() != spec["sha256"]:
        raise ValueError("Copied input checksum mismatch")
    # Read back from the mounted filesystem before publishing the completion marker.
    if target.stat().st_size != count or digest(target) != spec["sha256"]:
        raise ValueError("Destination read-back checksum mismatch")


def copy_inputs(source, target, spec, generated_files=None, *, workers=1):
    generated_files = generated_files or {}
    if workers not in (1, 8):
        raise ValueError("Unsupported input copy concurrency")

    def one(item):
        name, info = item
        copy_file(
            checked_path(source, name), checked_path(target, name), info, generated_files.get(name)
        )

    items = sorted(spec["files"].items())
    if workers == 1:
        for index, item in enumerate(items, 1):
            one(item)
            print(f"Progress: global input files {index}/{len(items)}", flush=True)
    else:
        # Cross-region object reads are latency-bound. Bound concurrency and memory;
        # each future retains the same source/size/hash/destination-readback checks.
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, item) for item in items]
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                future.result()
                print(f"Progress: global input files {index}/{len(items)}", flush=True)


def validate_report(result, sha, global_id, pod_id, *, training=False):
    r = result.get("report", {})
    if (
        result.get("status") != "passed"
        or r.get("kind") != ("global-training-export" if training else "global-export")
        or r.get("commit") != sha
        or r.get("global_volume_id") != global_id
        or not valid_id(global_id)
        or r.get("pod_id") != pod_id
        or r.get("source_volume_id") != SOURCE_ID
    ):
        raise RuntimeError("Global export identity mismatch")
    spec = manifest(r["provenance_json"], training=training)
    if (
        r.get("snapshot") != spec
        or r.get("snapshot_sha256") != hashlib.sha256(encoded(spec)).hexdigest()
    ):
        raise RuntimeError("Global snapshot differs from pinned inputs")
    return spec


def export_inputs(*, training=False):
    if not os.path.ismount("/data") or not os.path.ismount("/global"):
        raise RuntimeError("Both source and global volumes must be mounted")
    if os.environ.get("SPT_VOLUME_ID") != SOURCE_ID:
        raise RuntimeError("Wrong source volume")
    started = time.monotonic()
    source, target = Path("/data"), Path("/global") / (TRAIN_PREFIX if training else PREFIX)
    with (source / ".spleeterpt.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provenance_json = (source / "musdb18/provenance.json").read_text()
        spec = manifest(provenance_json, training=training)
        backup = evidence()["resume"]["backup"]

        def check_generation():
            if training:
                return
            if (
                json.loads((source / BACKUP / "latest.json").read_text())
                != {"generation": backup["generation"]}
                or json.loads(
                    (source / BACKUP / backup["generation"] / "manifest.json").read_text()
                )
                != backup
            ):
                raise ValueError("Source checkpoint generation changed")

        check_generation()
        marker = target / "snapshot.json"
        if marker.exists() and marker.read_bytes() != encoded(spec):
            raise ValueError("Refusing to replace a different completed snapshot")
        if marker.exists():
            for index, (name, info) in enumerate(spec["files"].items(), 1):
                existing = checked_path(target, name)
                if (
                    not existing.is_file()
                    or existing.stat().st_size != info["bytes"]
                    or digest(existing) != info["sha256"]
                ):
                    raise ValueError("Completed global snapshot is corrupted; refusing overwrite")
                print(f"Progress: global input files {index}/{len(spec['files'])}", flush=True)
        else:
            copy_inputs(source, target, spec, generated(provenance_json, training=training))
        check_generation()
        # No locks, chmod or atomic rename on object storage. This marker is written last.
        if not marker.exists():
            marker.write_bytes(encoded(spec))
        if marker.read_bytes() != encoded(spec):
            raise ValueError("Snapshot completion marker read-back failed")
    return {
        "kind": "global-training-export" if training else "global-export",
        "commit": os.environ["SPT_COMMIT"],
        "pod_id": os.environ["RUNPOD_POD_ID"],
        "source_volume_id": SOURCE_ID,
        "global_volume_id": os.environ["SPT_GLOBAL_VOLUME_ID"],
        "snapshot": spec,
        "snapshot_sha256": hashlib.sha256(encoded(spec)).hexdigest(),
        "provenance_json": provenance_json,
        "seconds": time.monotonic() - started,
    }


def stage_inputs(*, training=False):
    if not os.path.ismount("/global") or os.path.ismount("/data") or Path("/data").exists():
        raise RuntimeError("Global inputs require an empty local staging path")
    source = Path("/global") / (TRAIN_PREFIX if training else PREFIX)
    raw = (source / "snapshot.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != os.environ["SPT_SNAPSHOT_SHA256"]:
        raise RuntimeError("Global snapshot is missing, stale or changed")
    spec = manifest((source / "musdb18/provenance.json").read_text(), training=training)
    if raw != encoded(spec):
        raise RuntimeError("Global inputs differ from pinned validation inputs")
    if shutil.disk_usage("/").free < spec["bytes"] + 1024**3:
        raise RuntimeError("Insufficient local staging space")
    copy_inputs(source, Path("/data"), spec, workers=8 if training else 1)
    return {
        "mode": "global-staged",
        "global_volume_id": os.environ["SPT_GLOBAL_VOLUME_ID"],
        "source_volume_id": SOURCE_ID,
        "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "verified_audio_files": 500 if training else 50,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    report = export_inputs(training=args.training)
    validate_report(
        {"status": "passed", "report": report},
        os.environ["SPT_COMMIT"],
        os.environ["SPT_GLOBAL_VOLUME_ID"],
        os.environ["RUNPOD_POD_ID"],
        training=args.training,
    )
    args.output.write_text(json.dumps(report, indent=2))
