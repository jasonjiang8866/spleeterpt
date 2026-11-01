"""Stage only validation audio, run isolated models, then score on CPU and export."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from global_inputs import PREFIX, checked_path, copy_file, copy_inputs, encoded, evidence, manifest
from quality import ROOT, command, digest, download

sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_metrics import PROTOCOL, RATE, STEMS, VARIANTS, summarize, validate_baseline
from finetune_evaluation import validate_records, validation_inventory
from musdb_evaluation import atomic_json, score_variants
from probe_htdemucs import SHA256 as DEMUCS_SHA
from probe_htdemucs import URL as DEMUCS_URL


def stage():
    if not os.path.ismount("/global") or Path("/data").exists():
        raise ValueError("Expected existing Global mount and fresh local staging directory")
    source = Path("/global") / PREFIX
    raw = checked_path(source, "snapshot.json").read_bytes()
    provenance = checked_path(source, "musdb18/provenance.json").read_text()
    full = manifest(provenance)  # authenticates the exact original provenance bytes
    if raw != encoded(full):
        raise ValueError("Global completion marker differs from pinned validation snapshot")
    files = {k: v for k, v in full["files"].items() if k.startswith("musdb18/")}
    if len(files) != 51:
        raise ValueError("Expected 50 audio files plus original provenance only")
    size = sum(s["bytes"] for s in files.values())
    if shutil.disk_usage("/").free < size * 3 + 4 * 1024**3:
        raise ValueError("Insufficient local staging/output capacity")
    started = time.monotonic()
    copy_inputs(source, Path("/data"), {"files": files}, workers=8)
    return provenance, {
        "mode": "global-staged",
        "global_volume_id": os.environ["SPT_GLOBAL_VOLUME_ID"],
        "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "verified_audio_files": 50,
        "staged_files": len(files),
        "staged_bytes": size,
        "stage_seconds": time.monotonic() - started,
        "test_pool_used": False,
    }


def weights(folder):
    folder.mkdir()
    archive = download("4stems.tar.gz", folder)
    with tarfile.open(archive) as tar:
        tar.extractall(folder / "weights", filter="data")
    initial = folder / "initial.pt"
    command(
        sys.executable,
        "-m",
        "spleeterpt.cli",
        "convert",
        "--checkpoint",
        folder / "weights/model",
        "--stems",
        "4",
        "--output",
        initial,
    )
    if digest(initial) != evidence()["resume"]["initial_sha256"]:
        raise ValueError("Recreated official baseline checksum mismatch")
    demucs = folder / "955717e8-8726e21a.th"
    with urllib.request.urlopen(DEMUCS_URL, timeout=60) as response, demucs.open("wb") as stream:
        count = 0
        while chunk := response.read(1024**2):
            count += len(chunk)
            if count > 256 * 1024**2:
                raise ValueError("HTDemucs checkpoint exceeds download bound")
            stream.write(chunk)
    if digest(demucs) != DEMUCS_SHA:
        raise ValueError("HTDemucs checkpoint checksum mismatch")
    return dict(zip(VARIANTS, (initial, demucs)))


def load_audio(path, samples):
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != RATE or audio.shape != (samples, 2) or not np.isfinite(audio).all():
        raise ValueError("Invalid full-song PCM")
    return audio.T.copy()


def listening(out, index, track_id, samples, targets, variants, labels):
    import numpy as np
    import soundfile as sf

    count = RATE * 8
    start = (samples - count) // 2
    groups = {"reference": targets, **{label: variants[v] for label, v in labels.items()}}
    clips = {
        f"sample-{index + 1}/{group}-{stem}.wav": a[:, start : start + count]
        for group, stems in groups.items()
        for stem, a in stems.items()
    }
    gain = min(1.0, 0.95 / max(max(float(np.abs(a).max()) for a in clips.values()), 1e-12))
    for name, audio in clips.items():
        path = out / name
        path.parent.mkdir(exist_ok=True)
        sf.write(path, audio.T * gain, RATE, subtype="FLOAT")
    return {
        "id": track_id,
        "start_sample": start,
        "samples": count,
        "gain": gain,
        "files": sorted(clips),
    }


def run():
    from threadpoolctl import threadpool_limits

    started = time.monotonic()
    pod_id = os.environ["RUNPOD_POD_ID"]
    experiment = os.environ["SPT_EXPERIMENT"]
    if not re.fullmatch(r"[a-zA-Z0-9]+", pod_id) or not re.fullmatch(r"\d+-\d+", experiment):
        raise ValueError("Invalid run identity")
    provenance, storage = stage()
    pinned = json.loads((ROOT / "docs/results/musdb18-training-provenance.json").read_text())
    expected = validation_inventory(
        provenance, evidence()["resume"]["dataset"]["provenance_sha256"], pinned
    )
    work = Path("/workspace/benchmark")
    work.mkdir()
    checkpoints = weights(work / "assets")
    spec = {
        "tracks": [
            {
                "id": t,
                "name": t.removeprefix("train/"),
                "samples": s["samples"],
                "mixture_sha256": s["files"]["mixture"]["sha256"],
            }
            for t, s in sorted(expected.items())
        ],
        "weight_sha256": {v: digest(p) for v, p in checkpoints.items()},
    }
    atomic_json(work / "manifest.json", spec)
    measurements, smoke_measurements = {}, {}
    for smoke in (True, False):
        for variant in VARIANTS:
            output = work / (("smoke-" if smoke else "") + variant)
            command(
                sys.executable,
                "scripts/separator_benchmark.py",
                "--model",
                variant,
                "--weights",
                checkpoints[variant],
                "--manifest",
                work / "manifest.json",
                "--data",
                "/data/musdb18/train-pool",
                "--output",
                output,
                *(["--smoke"] if smoke else []),
            )
            measured = json.loads((output / "measurements.json").read_text())
            from benchmark_result import validate_measurement

            validate_measurement(measured, variant, expected, smoke=smoke)
            if smoke:
                smoke_measurements[variant] = measured
            else:
                measurements[variant] = measured
        if smoke:
            # Measure real full-song BSSEval too, before committing to the ten-song pass.
            first_id = min(expected)
            first = expected[first_id]
            targets = {
                s: load_audio(
                    Path("/data/musdb18/train-pool") / first_id.removeprefix("train/") / f"{s}.wav",
                    first["samples"],
                )
                for s in ("mixture", *STEMS)
            }
            variants = {
                v: {
                    s: load_audio(work / ("smoke-" + v) / "track-00" / f"{s}.wav", first["samples"])
                    for s in STEMS
                }
                for v in VARIANTS
            }
            tick = time.monotonic()
            with threadpool_limits(limits=2):
                score_variants(
                    targets, variants, STEMS, length_policy="strict", variant_names=VARIANTS
                )
            scoring_seconds = time.monotonic() - tick
            ratio = sum(t["samples"] for t in expected.values()) / first["samples"]
            projected = (
                sum(
                    m["tracks"][0]["repeats"][0]["end_to_end_seconds"] * 3
                    for m in smoke_measurements.values()
                )
                + scoring_seconds
            ) * ratio * 1.5 + 120
            smoke_report = {
                "kind": "benchmark-smoke",
                "measurements": smoke_measurements,
                "first_song_scoring_seconds": scoring_seconds,
                "projected_remaining_seconds_with_50pct_margin": projected,
                "available_worker_seconds": int(os.environ["SPT_EXPIRY"]) - time.time() - 180,
            }
            atomic_json(Path("/workspace/report.json"), smoke_report)
            if projected > smoke_report["available_worker_seconds"]:
                raise ValueError(
                    "One-song projection exceeds remaining lease; no full pass started"
                )
            del targets, variants
            for variant in VARIANTS:
                shutil.rmtree(work / ("smoke-" + variant))
    # Both model processes have exited. Scoring cannot consume GPU model memory.
    out = work / "export"
    out.mkdir()
    records, clips = [], []
    shuffled = list(VARIANTS)
    random.SystemRandom().shuffle(shuffled)
    labels = dict(zip(("A", "B"), shuffled))
    scoring_started = time.monotonic()
    with threadpool_limits(limits=2):
        for index, (track_id, track) in enumerate(sorted(expected.items())):
            targets = {}
            for stem in ("mixture", *STEMS):
                path = (
                    Path("/data/musdb18/train-pool")
                    / track_id.removeprefix("train/")
                    / f"{stem}.wav"
                )
                if digest(path) != track["files"][stem]["sha256"]:
                    raise ValueError("Target changed after staging")
                targets[stem] = load_audio(path, track["samples"])
            variants = {}
            for variant in VARIANTS:
                variants[variant] = {}
                for stem in STEMS:
                    path = work / variant / f"track-{index:02d}" / f"{stem}.wav"
                    if (
                        digest(path)
                        != measurements[variant]["tracks"][index]["files"][stem]["sha256"]
                    ):
                        raise ValueError("Inference output changed before scoring")
                    variants[variant][stem] = load_audio(path, track["samples"])
            record = {
                "id": track_id,
                "input_sha256": {s: x["sha256"] for s, x in track["files"].items()},
                "decoded_samples": track["samples"],
                "segment_start_sample": 0,
                **score_variants(
                    targets, variants, STEMS, length_policy="strict", variant_names=VARIANTS
                ),
            }
            records.append(record)
            atomic_json(out / f"track-{index:02d}.json", record)
            if index < 2:
                clips.append(
                    listening(out, index, track_id, track["samples"], targets, variants, labels)
                )
            del targets, variants
            print(f"Progress: evaluated validation songs {index + 1}/10", flush=True)
    validate_records(records, expected, variants=VARIANTS)
    summary = summarize(records)
    validate_baseline(
        summary, json.loads((ROOT / "docs/results/step90-validation-41196010747.json").read_text())
    )
    (out / "LISTENING.txt").write_text(
        "MUSDB18 AAC research excerpts; original source-specific licenses apply.\n"
        "https://sigsep.github.io/datasets/musdb.html ; https://zenodo.org/records/1117372\n"
        "First two sorted songs, centered eight seconds, common song gain.\n"
        "reference-mixture is input; reference stems are targets. A/B answers are in ANSWERS.json.\n"
        "Full-song scores use unscaled audio. No automated subjective listening verdict.\n"
    )
    atomic_json(out / "ANSWERS.json", labels)
    report = {
        "kind": "spleeter4-htdemucs",
        "commit": os.environ["SPT_COMMIT"],
        "pod_id": pod_id,
        "experiment": experiment,
        "storage": storage,
        "protocol": PROTOCOL,
        "protocol_sha256": hashlib.sha256(encoded(PROTOCOL)).hexdigest(),
        "provenance_json": provenance,
        "weight_sha256": spec["weight_sha256"],
        "measurements": measurements,
        "smoke": smoke_report,
        "records": records,
        "summary": summary,
        "listening": clips,
        "packages": {
            d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()
        },
        "scoring_seconds": time.monotonic() - scoring_started,
        "worker_seconds": time.monotonic() - started,
        "host": {
            "cpu_model": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "unknown",
            ),
            "visible_cpu_count": os.cpu_count(),
            "driver_versions": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                timeout=10,
            )
            .strip()
            .splitlines(),
            "sharing": "MIG partition; sibling tenant activity is not controlled or assumed absent",
        },
    }
    atomic_json(out / "report.json", report)
    files = {
        str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }
    bundle = Path("/workspace/checkpoint.tar")
    with tarfile.open(bundle, "w") as tar:
        for name in files:
            tar.add(out / name, arcname=name, recursive=False)
    report["bundle"] = {"bytes": bundle.stat().st_size, "sha256": digest(bundle), "files": files}
    # Immutable per-run objects, completion marker last. Never replace input snapshots.
    destination = Path("/global/benchmarks") / experiment
    if destination.exists():
        raise ValueError("Refusing to overwrite an existing benchmark export")
    copy_file(
        bundle,
        destination / "listening-and-metrics.tar",
        {k: report["bundle"][k] for k in ("bytes", "sha256")},
    )
    raw = encoded(report)
    copy_file(
        None,
        destination / "report.json",
        {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
        raw,
    )
    (destination / "complete.json").write_bytes(
        encoded(
            {
                "report_sha256": hashlib.sha256(raw).hexdigest(),
                "bundle_sha256": report["bundle"]["sha256"],
            }
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    atomic_json(args.output, result)
