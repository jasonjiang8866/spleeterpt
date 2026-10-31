"""Prepare full train-pool data, then exercise periodic storage across two Pods."""

import argparse
import concurrent.futures
import fcntl
import json
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from quality import command, digest, download

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/musdb18")
ARCHIVE_URL = "https://zenodo.org/api/records/1117372/files/musdb18.zip/content"
ARCHIVE_BYTES = 4684228845
ARCHIVE_SHA = "2765dd51fed264bbe2dac5c382f7120b7cf1932698fc67d100aa0f3811479152"


def archive_download(path):
    if path.exists() and path.stat().st_size == ARCHIVE_BYTES and digest(path) == ARCHIVE_SHA:
        return
    parts = path.with_suffix(".parts")
    parts.mkdir(exist_ok=True)
    chunk_size = 64 * 1024**2

    def part(start):
        end = min(start + chunk_size, ARCHIVE_BYTES) - 1
        target = parts / str(start)
        if target.exists() and target.stat().st_size == end - start + 1:
            return target
        req = urllib.request.Request(
            ARCHIVE_URL,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": "spleeterpt-data/1.0"},
        )
        with urllib.request.urlopen(req, timeout=120) as r, target.open("wb") as f:
            if (
                r.status != 206
                or r.headers.get("Content-Range") != f"bytes {start}-{end}/{ARCHIVE_BYTES}"
            ):
                raise RuntimeError("Official archive range mismatch")
            shutil.copyfileobj(r, f)
        if target.stat().st_size != end - start + 1:
            raise RuntimeError("Truncated archive part")
        return target

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(part, start): start for start in range(0, ARCHIVE_BYTES, chunk_size)}
        completed = {}
        for future in concurrent.futures.as_completed(futures):
            completed[futures[future]] = future.result()
            print(f"Progress: archive chunks {len(completed)}/{len(futures)}", flush=True)
        paths = [completed[start] for start in sorted(completed)]
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as f:
        for p in paths:
            with p.open("rb") as src:
                shutil.copyfileobj(src, f)
        f.flush()
        os.fsync(f.fileno())
    if digest(temporary) != ARCHIVE_SHA:
        raise RuntimeError("Official archive SHA-256 mismatch")
    os.replace(temporary, path)
    shutil.rmtree(parts)


def verify_data():
    provenance = json.loads((DATA / "provenance.json").read_text())
    pinned = json.loads((ROOT / "docs/results/musdb18-training-provenance.json").read_text())
    if (
        provenance.get("archive_sha256") != ARCHIVE_SHA
        or provenance.get("test_audio_extracted") is not False
        or len(provenance["tracks"]) != 100
    ):
        raise RuntimeError("Wrong full training dataset")
    expected = {x["id"]: x["source_sha256"] for x in pinned["tracks"]}
    if {x["id"]: x["source_sha256"] for x in provenance["tracks"]} != expected:
        raise RuntimeError("Source inventory differs from verified official release")
    for index, track in enumerate(provenance["tracks"], 1):
        folder = DATA / "train-pool" / Path(track["id"]).name.removesuffix(".stem.mp4")
        if set(track["files"]) != {"mixture", "drums", "bass", "other", "vocals"}:
            raise RuntimeError("Wrong stem inventory")
        for stem, spec in track["files"].items():
            p = folder / f"{stem}.wav"
            if p.stat().st_size != spec["bytes"] or digest(p) != spec["sha256"]:
                raise RuntimeError("Persistent audio checksum mismatch")
        print(f"Progress: verifying audio {index}/100", flush=True)
    splits = {}
    for split in ("train", "validation"):
        path = DATA / "manifests" / f"{split}.jsonl"
        if digest(path) != provenance["manifests"][path.name]:
            raise RuntimeError("Manifest checksum mismatch")
        tracks = [Path(json.loads(line)["track"]) for line in path.read_text().splitlines()]
        if any(p.parent != DATA / "train-pool" for p in tracks):
            raise RuntimeError("Manifest escaped persistent dataset")
        splits[split] = ["train/" + p.name for p in tracks]
    if splits != pinned["split_track_ids"]:
        raise RuntimeError("Train/validation split differs from recorded split")
    return {
        "archive_sha256": ARCHIVE_SHA,
        "provenance_sha256": digest(DATA / "provenance.json"),
        "train_tracks": 90,
        "validation_tracks": 10,
        "verified_audio_files": 500,
        "test_audio_extracted": False,
    }


def run(stage):
    import torch
    from training import phase

    from spleeterpt.backup import restore, validate
    from spleeterpt.checkpoint import read_checkpoint

    if not os.path.ismount("/data"):
        raise RuntimeError("Persistent mount is absent; refusing ephemeral fallback")
    started = time.monotonic()
    experiment = os.environ["SPT_EXPERIMENT"]
    if not re.fullmatch(r"[0-9]+-[0-9]+", experiment):
        raise RuntimeError("Invalid experiment identity")
    with Path("/data/.spleeterpt.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if stage == "prepare":
            if not (DATA / "provenance.json").exists():
                archive = Path("/data/musdb18.zip")
                archive_download(archive)
                command(
                    sys.executable,
                    "scripts/prepare_musdb_training.py",
                    "--archive",
                    archive,
                    "--output",
                    DATA,
                )
            dataset = verify_data()
            return base_report(stage, started, dataset)
        dataset = verify_data()
        if torch.__version__ != "2.8.0+cu128" or not torch.cuda.is_available():
            raise RuntimeError("Expected stable CUDA runtime")
        backup = Path("/data/experiments") / experiment / "backups"
        meta_path = backup.parent / "experiment.json"
        out = Path("/workspace/persistent-pilot/run")
        restored = None
        if stage == "train":
            if backup.parent.exists():
                raise FileExistsError("Experiment exists; use resume")
            backup.parent.mkdir(parents=True)
            folder = Path("/workspace/persistent-pilot")
            folder.mkdir()
            weights = folder / "weights"
            weights.mkdir()
            archive = download("4stems.tar.gz", folder)
            with tarfile.open(archive) as tar:
                tar.extractall(weights, filter="data")
            initial = folder / "initial.pt"
            command(
                sys.executable,
                "-m",
                "spleeterpt.cli",
                "convert",
                "--checkpoint",
                weights / "model",
                "--stems",
                "4",
                "--output",
                initial,
            )
            meta = {
                "commit": os.environ["SPT_COMMIT"],
                "source_pod": os.environ["RUNPOD_POD_ID"],
                "dataset": dataset,
                "initial_sha256": digest(initial),
            }
            meta_path.write_text(json.dumps(meta))
            with meta_path.open("rb") as f:
                os.fsync(f.fileno())
            extra = ["--init", str(initial), "--max-steps", "10"]
        else:
            meta = json.loads(meta_path.read_text())
            if (
                meta["dataset"] != dataset
                or meta["commit"] != os.environ["SPT_COMMIT"]
                or meta["source_pod"] == os.environ["RUNPOD_POD_ID"]
            ):
                raise RuntimeError("Resume must use same data/commit on a different Pod")
            checkpoint, restored = restore(backup, out)
            cp = read_checkpoint(checkpoint)
            if [cp[k] for k in ("step", "epoch", "next_batch")] != [10, 0, 10]:
                raise RuntimeError("Expected interrupted step 10")
            del cp
            extra = ["--resume", str(checkpoint)]
        args = [
            "train",
            "--config",
            str(ROOT / "configs/4stems.json"),
            "--train-manifest",
            str(DATA / "manifests/train.jsonl"),
            "--val-manifest",
            str(DATA / "manifests/validation.jsonl"),
            "--output",
            str(out),
            "--backup-dir",
            str(backup),
            "--device",
            "cuda",
            "--precision",
            "fp32",
            "--batch-size",
            "1",
            "--epochs",
            "1",
            "--crops-per-track",
            "1",
            "--validation-crops",
            "1",
            "--workers",
            "0",
            "--save-every",
            "10",
            "--lr",
            "1e-5",
            "--seed",
            "17",
            "--deterministic",
            *extra,
        ]
        timing = out.parent / "timing.json"
        phase(args, timing)
        _, saved = validate(backup)
        cp = read_checkpoint(out / "last.pt")
        metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
        r = base_report(stage, started, dataset)
        r.update(
            source_pod=meta["source_pod"],
            cursor=[cp[k] for k in ("step", "epoch", "next_batch")],
            optimizer_steps=sorted({int(v["step"]) for v in cp["optimizer"]["state"].values()}),
            weights_finite=all(torch.isfinite(t).all().item() for t in cp["model"].values()),
            metrics=metrics,
            backup=saved,
            restored=restored,
            initial_sha256=meta["initial_sha256"],
            timing=json.loads(timing.read_text()),
            config=cp["config"],
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
        )
        return r


def base_report(stage, started, dataset):
    return {
        "kind": "persistent-pilot",
        "stage": stage,
        "commit": os.environ["SPT_COMMIT"],
        "pod_id": os.environ["RUNPOD_POD_ID"],
        "volume_id": os.environ["SPT_VOLUME_ID"],
        "experiment": os.environ["SPT_EXPERIMENT"],
        "dataset": dataset,
        "seconds": time.monotonic() - started,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["prepare", "train", "resume"], required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = run(a.stage)
    a.output.write_text(json.dumps(report, indent=2))
    from persistent_result import validate_persistent

    validate_persistent(
        {"status": "passed", "report": report},
        os.environ["SPT_COMMIT"],
        "persistent-" + a.stage,
        os.environ["SPT_VOLUME_ID"],
        os.environ["RUNPOD_POD_ID"],
    )
