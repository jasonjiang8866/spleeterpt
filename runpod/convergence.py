"""Bounded two-LR pilot with durable milestones and fresh-process continuation."""

import argparse
import gc
import importlib.metadata
import json
import math
import os
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

from frozen import fingerprint
from global_inputs import copy_inputs, encoded, evidence, stage_inputs
from persistent import DATA, ROOT, verify_data
from quality import command, digest, download
from training import equal_state

sys.path.insert(0, str(ROOT / "scripts"))
from convergence_evaluation import ARMS, MILESTONES, PROTOCOL, VARIANTS, summarize
from evaluate_preview import si_sdr
from finetune_evaluation import RATE, STEMS, validation_inventory
from musdb_evaluation import atomic_json, score_variants

CHECKPOINT_FILES = ("last.pt", "config.json", "arguments.json", "metrics.jsonl", "timing.json")


def publish(folder, experiment, arm, step):
    spec = {
        "files": {
            n: {"bytes": (folder / n).stat().st_size, "sha256": digest(folder / n)}
            for n in CHECKPOINT_FILES
        }
    }
    if sum(s["bytes"] for s in spec["files"].values()) > 512 * 1024**2:
        raise ValueError("Checkpoint exceeds 512 MiB milestone allowance")
    target = Path("/global/convergence-runs") / experiment / arm / str(step)
    if target.exists():
        raise FileExistsError("Immutable checkpoint destination exists")
    copy_inputs(folder, target, spec)
    marker = target / "complete.json"
    marker.write_bytes(encoded(spec))
    if marker.read_bytes() != encoded(spec):
        raise ValueError("Durable completion marker readback mismatch")
    return {
        "prefix": str(target.relative_to("/global")),
        **spec,
        "manifest_sha256": digest(marker),
        "readback_verified": True,
    }


def restore(descriptor, destination):
    """Force a new local copy, not a cache hit on the training output."""
    if destination.exists():
        raise FileExistsError("Restore destination must be fresh")
    source = Path("/global") / descriptor["prefix"]
    if digest(source / "complete.json") != descriptor["manifest_sha256"]:
        raise ValueError("Global marker changed before resume")
    copy_inputs(source, destination, {"files": {"last.pt": descriptor["files"]["last.pt"]}})
    return destination / "last.pt"


def coverage(dataset, epochs):
    """Use the loader's actual crop rule; intervals describe coverage, not shuffle order."""
    records = []
    previous = dataset.epoch
    for i, (_, _, frames) in enumerate(dataset.records):
        starts = []
        for epoch in range(epochs):
            dataset.epoch = epoch
            starts.extend(
                dataset.crop_start(i * dataset.samples_per_track + j)
                for j in range(dataset.samples_per_track)
            )
        covered, end = 0, 0
        for start in sorted(starts):
            stop = min(frames, start + dataset.config.segment_samples)
            covered += max(0, stop - max(start, end))
            end = max(end, stop)
        records.append(
            {
                "id": "train/" + dataset.tracks[i].name,
                "samples": frames,
                "starts": starts,
                "covered_samples": covered,
            }
        )
    dataset.epoch = previous
    return {"epochs": epochs, "crop_samples": dataset.config.segment_samples, "tracks": records}


def budget_projection(phase_seconds, baseline_seconds, remaining_steps, remaining_scores):
    # Measured complete phase includes startup, checkpoint saves and epoch validation.
    estimate = 1.25 * (phase_seconds * remaining_steps / 90 + baseline_seconds * remaining_scores)
    return estimate + 240  # SI curves, Global I/O, report export; hard lease is independent.


def run(report_path):
    import numpy as np
    import soundfile as sf
    import torch
    from benchmark_dependencies import check_scoring_runtime
    from gpu_profile import MIG24, validate_runtime
    from threadpoolctl import threadpool_limits

    from spleeterpt.audio import separate
    from spleeterpt.checkpoint import load_model, read_checkpoint
    from spleeterpt.config import ModelConfig
    from spleeterpt.data import StemDataset

    started = time.monotonic()
    check_scoring_runtime()
    experiment = os.environ["SPT_EXPERIMENT"]
    if not re.fullmatch(r"[0-9]+-[0-9]+", experiment):
        raise ValueError("Invalid experiment")
    storage = stage_inputs(training=True)
    dataset = verify_data()
    if dataset != evidence()["resume"]["dataset"]:
        raise ValueError("Training data changed")
    if torch.__version__ != "2.8.0+cu128" or not torch.cuda.is_available():
        raise RuntimeError("Expected pinned CUDA runtime")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    runtime = {
        "requested_gpu": os.environ["SPT_GPU_ID"],
        "name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    validate_runtime(runtime, MIG24)
    out = Path("/workspace/convergence-pilot")
    out.mkdir()
    results = out / "evaluation"
    results.mkdir()
    archive = download("4stems.tar.gz", out)
    with tarfile.open(archive) as tar:
        tar.extractall(out / "weights", filter="data")
    initial = out / "initial.pt"
    command(
        sys.executable,
        "-m",
        "spleeterpt.cli",
        "convert",
        "--checkpoint",
        out / "weights/model",
        "--stems",
        "4",
        "--output",
        initial,
    )
    if digest(initial) != evidence()["resume"]["initial_sha256"]:
        raise ValueError("Official initialization changed")
    model = load_model(initial)
    state = model.state_dict()
    buffers = set(dict(model.named_buffers()))
    bn = {k for k in buffers if k.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    affine = {k.rsplit(".", 1)[0] + "." + s for k in bn for s in ("weight", "bias")}
    parameters = set(state) - buffers
    initial_bn, initial_affine, initial_parameters = (
        fingerprint(state, keys) for keys in (bn, affine, parameters)
    )
    del model, state
    provenance_json = (DATA / "provenance.json").read_text()
    pinned = json.loads((ROOT / "docs/results/musdb18-training-provenance.json").read_text())
    expected = validation_inventory(provenance_json, dataset["provenance_sha256"], pinned)
    train = StemDataset(
        DATA / "manifests/train.jsonl",
        ModelConfig.load(ROOT / "configs/4stems.json"),
        training=True,
        seed=17,
    )
    report = {
        "kind": "convergence-pilot",
        "commit": os.environ["SPT_COMMIT"],
        "pod_id": os.environ["RUNPOD_POD_ID"],
        "experiment": experiment,
        "storage": storage,
        "dataset": dataset,
        "protocol": PROTOCOL,
        "initial_sha256": digest(initial),
        "initial_bn_fingerprint": initial_bn,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_runtime": runtime,
        "museval": importlib.metadata.version("museval"),
        "provenance_json": provenance_json,
        "coverage": coverage(train, 10),
        "arms": {},
        "curves": {},
        "records": [],
        "budget_projections": [],
        "weights_unchanged": True,
        "complete": False,
    }

    def persist():
        report["seconds"] = time.monotonic() - started
        atomic_json(report_path, report)

    def quality(checkpoint, label, *, bsseval):
        tick = time.monotonic()
        model = load_model(checkpoint, "cuda")
        before = fingerprint(model.state_dict(), model.state_dict())
        records, curve = [], []
        torch.cuda.reset_peak_memory_stats()
        with threadpool_limits(limits=2):
            for index, tid in enumerate(sorted(expected)):
                targets = {}
                for stem in ("mixture", *STEMS):
                    path = DATA / "train-pool" / tid.removeprefix("train/") / (stem + ".wav")
                    if digest(path) != expected[tid]["files"][stem]["sha256"]:
                        raise ValueError("Evaluation input changed")
                    audio, rate = sf.read(path, dtype="float32", always_2d=True)
                    if rate != RATE or audio.shape != (expected[tid]["samples"], 2):
                        raise ValueError("Full-song shape mismatch")
                    targets[stem] = audio.T.copy()
                output = {
                    s: t.numpy()
                    for s, t in separate(model, torch.from_numpy(targets["mixture"])).items()
                }
                if set(output) != set(STEMS) or any(
                    output[s].shape != targets[s].shape or not np.isfinite(output[s]).all()
                    for s in STEMS
                ):
                    raise ValueError("Invalid full-song inference")
                scores = {s: si_sdr(targets[s], output[s]) for s in STEMS}
                if any(x is None or not math.isfinite(x) for x in scores.values()):
                    raise ValueError("Nonfinite SI-SDR curve")
                curve.append({"id": tid, "samples": expected[tid]["samples"], "scores": scores})
                if bsseval:
                    scored = score_variants(
                        targets,
                        {label: output},
                        STEMS,
                        length_policy="strict",
                        variant_names=(label,),
                    )
                    records.append(
                        {
                            "id": tid,
                            "input_sha256": {
                                s: x["sha256"] for s, x in expected[tid]["files"].items()
                            },
                            "decoded_samples": expected[tid]["samples"],
                            "segment_start_sample": 0,
                            **scored,
                        }
                    )
                    # Keep only two centered snippets locally; shared gain applied after all models.
                    if index < 2:
                        count = RATE * 8
                        start = (expected[tid]["samples"] - count) // 2
                        groups = {label: output}
                        if label == VARIANTS[0]:
                            groups["reference"] = targets
                        for group, stems in groups.items():
                            for stem, a in stems.items():
                                path = results / f"sample-{index + 1}/{group}-{stem}.wav"
                                path.parent.mkdir(exist_ok=True)
                                sf.write(path, a[:, start : start + count].T, RATE, subtype="FLOAT")
                print(f"Progress: convergence {label} songs {index + 1}/10", flush=True)
        if before != fingerprint(model.state_dict(), model.state_dict()):
            raise ValueError("Evaluation changed weights")
        peak = torch.cuda.max_memory_allocated()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "tracks": curve,
            "seconds": time.monotonic() - tick,
            "peak_allocated_bytes": peak,
        }, records

    report["curves"][VARIANTS[0]], baseline = quality(initial, VARIANTS[0], bsseval=True)
    report["records"] = baseline
    persist()
    rng, signatures = {}, {}
    completed_steps, completed_scores = 0, 0
    for arm, lr in ARMS.items():
        folder = out / arm
        proofs = []
        report["arms"][arm] = proofs
        resume = None
        for step in MILESTONES:
            epoch = step // 90
            timing = out / f"{arm}-{step}-timing.json"
            argv = [
                "train",
                "--config",
                str(ROOT / "configs/4stems.json"),
                "--train-manifest",
                str(DATA / "manifests/train.jsonl"),
                "--val-manifest",
                str(DATA / "manifests/validation.jsonl"),
                "--output",
                str(folder),
                "--device",
                "cuda",
                "--precision",
                "fp32",
                "--batch-size",
                "1",
                "--epochs",
                str(epoch),
                "--crops-per-track",
                "1",
                "--validation-crops",
                "4",
                "--workers",
                "0",
                "--save-every",
                "90",
                "--lr",
                str(lr),
                "--seed",
                "17",
                "--deterministic",
                "--freeze-bn-stats",
                "--resume" if resume else "--init",
                str(resume or initial),
            ]
            resume_sha = digest(resume) if resume else None
            command(
                sys.executable, ROOT / "runpod/training.py", "--phase-output", timing, "--", *argv
            )
            shutil.copyfile(timing, folder / "timing.json")
            cp = read_checkpoint(folder / "last.pt")
            state = cp["model"]
            proof = {
                "cursor": [cp[k] for k in ("step", "epoch", "next_batch")],
                "optimizer_steps": sorted(
                    {int(v["step"]) for v in cp["optimizer"]["state"].values()}
                ),
                "finite": all(torch.isfinite(t).all().item() for t in state.values()),
                "bn_buffers_unchanged": fingerprint(state, bn) == initial_bn,
                "bn_affine_changed": fingerprint(state, affine) != initial_affine,
                "parameters_changed": fingerprint(state, parameters) != initial_parameters,
                "bn_fingerprint": fingerprint(state, bn),
                "signature": cp["signature"],
                "timing": json.loads(timing.read_text()),
                "metrics": [
                    json.loads(x) for x in (folder / "metrics.jsonl").read_text().splitlines()
                ],
                "resume_sha256": resume_sha,
            }
            if (
                proof["cursor"] != [step, epoch, 0]
                or proof["optimizer_steps"] != [step]
                or not all(
                    proof[k]
                    for k in (
                        "finite",
                        "bn_buffers_unchanged",
                        "bn_affine_changed",
                        "parameters_changed",
                    )
                )
            ):
                raise ValueError("Training acceptance failed")
            proof["checkpoint"] = publish(folder, experiment, arm, step)
            if step != MILESTONES[-1]:
                resume = restore(proof["checkpoint"], out / f"restore-{arm}-{step}")
                restored = read_checkpoint(resume)
                proof["global_restore_equal"] = equal_state(cp, restored)
                if not proof["global_restore_equal"]:
                    raise ValueError("Global round-trip changed training state")
                del restored
            else:
                rng[arm] = (cp["rng_cpu"], cp["rng_cuda"])
                signatures[arm] = {k: v for k, v in cp["signature"].items() if k != "lr"}
            del cp, state
            gc.collect()
            proofs.append(proof)
            completed_steps += step - (MILESTONES[len(proofs) - 2] if len(proofs) > 1 else 0)
            persist()
            if arm == "lr1e5" and step == 90:
                projection = budget_projection(
                    proof["timing"]["seconds"],
                    report["curves"][VARIANTS[0]]["seconds"],
                    1800 - completed_steps,
                    2,
                )
                remaining = int(os.environ["SPT_EXPIRY"]) - time.time() - 300
                report["budget_projections"].append(
                    {
                        "remaining_seconds": remaining,
                        "projected_seconds": projection,
                        "accepted": projection < remaining,
                    }
                )
                persist()
                if projection >= remaining:
                    raise RuntimeError("Measured runtime cannot fit remaining bounded lease")
            label = f"{arm}_step{step}"
            if step == 900:
                projection = (
                    1.25 * report["curves"][VARIANTS[0]]["seconds"] * (2 - completed_scores) + 120
                )
                remaining = int(os.environ["SPT_EXPIRY"]) - time.time() - 180
                if projection >= remaining:
                    raise RuntimeError("Insufficient remaining lease for complete final scoring")
            report["curves"][label], records = quality(
                folder / "last.pt", label, bsseval=step == 900
            )
            if records:
                for existing, extra in zip(report["records"], records, strict=True):
                    if existing["id"] != extra["id"]:
                        raise ValueError("Mismatched paired song")
                    for key in ("scores", "output_samples"):
                        existing[key].update(extra[key])
                completed_scores += 1
            persist()
            print(f"Progress: convergence {arm} step {step} complete", flush=True)
    report["matched_rng"] = equal_state(rng["lr1e5"], rng["lr3e6"])
    if not report["matched_rng"] or signatures["lr1e5"] != signatures["lr3e6"]:
        raise ValueError("Unmatched training inputs/RNG consumption")
    report["summary"] = summarize(report["records"], expected)
    report["listening"] = []
    for i, tid in enumerate(sorted(expected)[:2]):
        paths = sorted((results / f"sample-{i + 1}").glob("*.wav"))
        clips = {p: sf.read(p, dtype="float32", always_2d=True)[0] for p in paths}
        gain = min(1.0, 0.95 / max(max(float(np.abs(a).max()) for a in clips.values()), 1e-12))
        for p, audio in clips.items():
            sf.write(p, audio * gain, RATE, subtype="FLOAT")
        report["listening"].append(
            {
                "id": tid,
                "samples": RATE * 8,
                "start_sample": (expected[tid]["samples"] - RATE * 8) // 2,
                "gain": gain,
                "files": [str(p.relative_to(results)) for p in paths],
            }
        )
    for i, record in enumerate(report["records"]):
        atomic_json(results / f"track-{i:02d}.json", record)
    (results / "LISTENING.txt").write_text(
        "MUSDB18 research evaluation excerpts; original source licenses apply.\n"
        "Source: https://zenodo.org/records/1117372 and https://sigsep.github.io/datasets/musdb.html\n"
        "Educational use only. Artist/title and centered 8-second intervals in report.json.\n"
        "One shared gain per song. reference-* are targets; reference-mixture is input.\n"
        "official_converted is baseline; lr1e5_step900 and lr3e6_step900 are frozen-BN fine-tunes.\n"
        "Scores use unscaled full recordings. No subjective listening verdict claimed.\n"
    )
    report["complete"] = True
    persist()
    atomic_json(results / "report.json", report)
    files = {
        str(p.relative_to(results)): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(results.rglob("*"))
        if p.is_file()
    }
    bundle = Path("/workspace/checkpoint.tar")
    with tarfile.open(bundle, "w") as tar:
        for name in files:
            tar.add(results / name, arcname=name, recursive=False)
    report["bundle"] = {"bytes": bundle.stat().st_size, "sha256": digest(bundle), "files": files}
    # Do not update seconds after report.json was archived: export validator checks equality.
    atomic_json(report_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output)
    from convergence_result import validate

    validate(
        {"status": "passed", "report": report},
        os.environ["SPT_COMMIT"],
        os.environ["SPT_GLOBAL_VOLUME_ID"],
        os.environ["RUNPOD_POD_ID"],
        os.environ["SPT_EXPERIMENT"],
    )
