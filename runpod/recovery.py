"""Checkpoint export on Pod A; exact continuation on a newly provisioned Pod B."""

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path

from quality import command, digest, download
from training import prepare_preview

ROOT = Path(__file__).resolve().parents[1]
FOLDER = Path("/workspace/training-recovery")


def state_hash(value):
    """Hash types, structure and exact tensor bytes, independently of torch.save containers."""
    import torch

    h = hashlib.sha256()

    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(json.dumps(["tensor", str(x.dtype), list(x.shape)]).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, dict):
            h.update(b"dict[")
            for k in sorted(x, key=lambda k: (type(k).__name__, repr(k))):
                visit(k)
                visit(x[k])
            h.update(b"]")
        elif isinstance(x, (list, tuple)):
            h.update(type(x).__name__.encode() + b"[")
            for item in x:
                visit(item)
            h.update(b"]")
        else:
            h.update(json.dumps([type(x).__name__, x], allow_nan=False).encode())

    visit(value)
    return h.hexdigest()


def fingerprints(checkpoint):
    from training_result import STATE_KEYS

    return {key: state_hash(checkpoint[key]) for key in sorted(STATE_KEYS)}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def pack(out, metadata, *, complete):
    from recovery_result import validate_bundle

    (out / "recovery.json").write_text(json.dumps(metadata, indent=2))
    names = ["last.pt", "metrics.jsonl", "recovery.json"] + (["best.pt"] if complete else [])
    files = {n: {"sha256": digest(out / n), "bytes": (out / n).stat().st_size} for n in names}
    path = Path("/workspace/checkpoint.tar")
    with tarfile.open(path, "w") as tar:
        for name in names:
            tar.add(out / name, arcname=name, recursive=False)
    descriptor = {"sha256": digest(path), "bytes": path.stat().st_size, "files": files}
    validate_bundle(path, descriptor, complete=complete)
    return descriptor


def run(stage):
    import torch
    from recovery_result import validate_bundle

    from spleeterpt.audio import read_audio, separate
    from spleeterpt.checkpoint import load_model, read_checkpoint

    if torch.__version__ != "2.8.0+cu128" or not torch.cuda.is_available():
        raise RuntimeError("Expected stable torch 2.8.0+cu128 on CUDA")
    torch.set_num_threads(4)
    started = time.monotonic()
    spec, decoded = prepare_preview(FOLDER)
    out = FOLDER / "run"
    common = [
        "train",
        "--config",
        str(ROOT / "configs/4stems.json"),
        "--train-manifest",
        str(FOLDER / "train.jsonl"),
        "--val-manifest",
        str(FOLDER / "validation.jsonl"),
        "--device",
        "cuda",
        "--precision",
        "fp32",
        "--batch-size",
        "1",
        "--workers",
        "0",
        "--crops-per-track",
        "2",
        "--validation-crops",
        "1",
        "--save-every",
        "1",
        "--epochs",
        "2",
        "--lr",
        "1e-5",
        "--seed",
        "17",
        "--deterministic",
    ]
    timings = {}

    def train(label, dest, extra):
        timing = FOLDER / f"{label}.json"
        command(
            sys.executable,
            ROOT / "runpod/training.py",
            "--phase-output",
            timing,
            "--",
            *common,
            "--output",
            dest,
            *extra,
        )
        timings[label] = json.loads(timing.read_text())

    if stage == "export":
        weights = FOLDER / "weights"
        weights.mkdir()
        archive = download("4stems.tar.gz", FOLDER)
        with tarfile.open(archive) as tar:
            tar.extractall(weights, filter="data")
        initial = FOLDER / "initial.pt"
        command(
            sys.executable,
            "-m",
            "spleeterpt.cli",
            "convert",
            "--checkpoint",
            weights / "model",
            "--stems",
            4,
            "--output",
            initial,
        )
        baseline = FOLDER / "continuous"
        train("continuous", baseline, ["--init", str(initial)])
        train("interrupted", out, ["--init", str(initial), "--max-steps", "3"])
        # Hash audio contents, not only mtime, before preserving metadata used by resume.
        data_files = {
            str(p.relative_to(FOLDER)): {
                "sha256": digest(p),
                "bytes": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns,
            }
            for p in sorted(FOLDER.glob("*/*/*.wav"))
        }
        final = read_checkpoint(baseline / "last.pt")
        metadata = {
            "commit": os.environ["SPT_COMMIT"],
            "source_pod": os.environ["RUNPOD_POD_ID"],
            "assets": spec,
            "decoded": decoded,
            "data_files": data_files,
            "expected_last": fingerprints(final),
            "expected_best": fingerprints(read_checkpoint(baseline / "best.pt")),
            "expected_metrics": rows(baseline / "metrics.jsonl"),
            "initial_sha256": digest(initial),
        }
        del final
    else:
        incoming = Path("/workspace/restore.tar")
        deadline = min(
            time.monotonic() + 300,
            time.monotonic() + int(os.environ["SPT_EXPIRY"]) - time.time() - 180,
        )
        while not incoming.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Restore bundle was not delivered")
            time.sleep(1)
        descriptor = json.loads(os.environ["SPT_RESTORE_DESCRIPTOR"])
        validate_bundle(incoming, descriptor, complete=False)
        out.mkdir()
        with tarfile.open(incoming) as tar:
            tar.extractall(out, filter="data")
        metadata = json.loads((out / "recovery.json").read_text())
        if (
            metadata["commit"] != os.environ["SPT_COMMIT"]
            or metadata["assets"] != spec
            or metadata["decoded"] != decoded
        ):
            raise ValueError("Restored code or audio provenance differs")
        if metadata["source_pod"] == os.environ["RUNPOD_POD_ID"]:
            raise ValueError("Recovery must run on a different Pod")
        actual_paths = {str(p.relative_to(FOLDER)) for p in FOLDER.glob("*/*/*.wav")}
        if set(metadata["data_files"]) != actual_paths:
            raise ValueError("Restored audio inventory differs")
        for name, item in metadata["data_files"].items():
            path = FOLDER / name
            if digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise ValueError("Restored audio contents differ")
            os.utime(path, ns=(item["mtime_ns"], item["mtime_ns"]))
        checkpoint = read_checkpoint(out / "last.pt")
        if [checkpoint["step"], checkpoint["epoch"], checkpoint["next_batch"]] != [3, 0, 3]:
            raise ValueError("Wrong restore cursor")
        if rows(out / "metrics.jsonl") != metadata["expected_metrics"][:3]:
            raise ValueError("Interrupted metrics mismatch")
        del checkpoint
        train("resumed", out, ["--resume", str(out / "last.pt")])
    last = read_checkpoint(out / "last.pt")
    metric_rows = rows(out / "metrics.jsonl")
    final_hashes = fingerprints(last) if stage == "resume" else None
    best_hashes = fingerprints(read_checkpoint(out / "best.pt")) if stage == "resume" else None
    inference_ok = None
    if stage == "resume":
        model = load_model(out / "best.pt", "cuda")
        track = Path(json.loads((FOLDER / "validation.jsonl").read_text())["track"])
        audio = read_audio(track / "mixture.wav", 44100)
        estimates = separate(model, audio)
        inference_ok = set(estimates) == set(model.config.instruments) and all(
            x.shape == audio.shape and torch.isfinite(x).all().item() for x in estimates.values()
        )
    report = {
        "kind": "pod-recovery",
        "stage": stage,
        "commit": os.environ["SPT_COMMIT"],
        "pod_id": os.environ["RUNPOD_POD_ID"],
        "source_pod": metadata["source_pod"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "config": last["config"],
        "assets": spec,
        "decoded": decoded,
        "cursor": [last["step"], last["epoch"], last["next_batch"]],
        "metrics": metric_rows,
        "expected_metrics": metadata["expected_metrics"],
        "expected_last": metadata["expected_last"],
        "expected_best": metadata["expected_best"],
        "actual_last": final_hashes,
        "actual_best": best_hashes,
        "optimizer_steps": sorted(
            {int(v["step"].item()) for v in last["optimizer"]["state"].values()}
        ),
        "weights_finite": all(torch.isfinite(v).all().item() for v in last["model"].values()),
        "inference_finite": inference_ok,
        "timings": timings,
        "bundle": pack(out, metadata, complete=stage == "resume"),
        "seconds": time.monotonic() - started,
    }
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["export", "resume"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = run(args.stage)
    args.output.write_text(json.dumps(report, indent=2))
    from recovery_result import validate_recovery

    validate_recovery({"status": "passed", "report": report}, os.environ["SPT_COMMIT"], args.stage)
