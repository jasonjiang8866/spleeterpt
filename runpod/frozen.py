"""Matched ordinary/frozen-BN pilot, local POSIX checkpoints, immutable global export."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import tarfile
import time
from pathlib import Path

from global_inputs import copy_inputs, encoded, evidence, stage_inputs
from persistent import DATA, ROOT, verify_data
from quality import command, digest, download
from training import phase

sys.path.insert(0, str(ROOT / "scripts"))
from finetune_evaluation import RATE, STEMS, validation_inventory
from frozen_bn_evaluation import PROTOCOL, VARIANTS, summarize
from musdb_evaluation import atomic_json, score_variants


def fingerprint(state, names):
    h = hashlib.sha256()
    for k in sorted(names):
        t = state[k].detach().cpu().contiguous()
        h.update(f"{k}:{t.dtype}:{tuple(t.shape)}:".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def publish_arm(folder, experiment, arm):
    """Publish only a completed immutable checkpoint, never train on object storage."""
    names = ("last.pt", "config.json", "arguments.json", "metrics.jsonl", "timing.json")
    spec = {
        "files": {
            name: {"bytes": (folder / name).stat().st_size, "sha256": digest(folder / name)}
            for name in names
        }
    }
    if sum(x["bytes"] for x in spec["files"].values()) > 512 * 1024**2:
        raise ValueError("Checkpoint export exceeds 512 MiB per arm")
    target = Path("/global/frozen-bn-runs") / experiment / arm
    marker = target / "complete.json"
    if target.exists():
        raise FileExistsError("Refusing to overwrite an existing experiment")
    copy_inputs(folder, target, spec)
    marker.write_bytes(encoded(spec))
    if marker.read_bytes() != encoded(spec):
        raise ValueError("Checkpoint export completion marker mismatch")
    return {
        "prefix": str(target.relative_to("/global")),
        **spec,
        "manifest_sha256": digest(marker),
        "readback_verified": True,
    }


def run():
    import numpy as np
    import soundfile as sf
    import torch
    from threadpoolctl import threadpool_limits
    from spleeterpt.audio import separate
    from spleeterpt.checkpoint import load_model, read_checkpoint

    started = time.monotonic()
    experiment = os.environ["SPT_EXPERIMENT"]
    if not re.fullmatch(r"[0-9]+-[0-9]+", experiment):
        raise ValueError("Invalid experiment")
    storage = stage_inputs(training=True)
    dataset = verify_data()
    if dataset != evidence()["resume"]["dataset"]:
        raise ValueError("Training data changed")
    if torch.__version__ != "2.8.0+cu128" or not torch.cuda.is_available():
        raise RuntimeError("Expected pinned CUDA runtime")
    from gpu_profile import MIG24, validate_runtime

    runtime = {
        "requested_gpu": os.environ["SPT_GPU_ID"],
        "name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    validate_runtime(runtime, MIG24)
    out = Path("/workspace/frozen-pilot")
    out.mkdir()
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
    initial_model = load_model(initial)
    initial_state = initial_model.state_dict()
    buffers = set(dict(initial_model.named_buffers()))
    bn_keys = {
        k for k in buffers if k.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }
    affine = {k.rsplit(".", 1)[0] + "." + suffix for k in bn_keys for suffix in ("weight", "bias")}
    if not bn_keys or not affine <= initial_state.keys():
        raise ValueError("Expected BN affine and running tensors")
    initial_bn = fingerprint(initial_state, bn_keys)
    initial_affine = fingerprint(initial_state, affine)
    initial_params = fingerprint(initial_state, set(initial_state) - buffers)
    arms, rng, signatures = {}, {}, {}
    for arm in ("normal", "frozen"):
        folder = out / arm
        folder.mkdir()
        args = [
            "train",
            "--config",
            str(ROOT / "configs/4stems.json"),
            "--train-manifest",
            str(DATA / "manifests/train.jsonl"),
            "--val-manifest",
            str(DATA / "manifests/validation.jsonl"),
            "--output",
            str(folder),
            "--backup-dir",
            str(out / (arm + "-backups")),
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
            "--init",
            str(initial),
        ]
        if arm == "frozen":
            args.append("--freeze-bn-stats")
        # phase writes timing only after training exits; the output directory starts empty.
        timing = out / (arm + "-timing.json")
        phase(args, timing)
        (folder / "timing.json").write_bytes(timing.read_bytes())
        cp = read_checkpoint(folder / "last.pt")
        state = cp["model"]
        final_bn = fingerprint(state, bn_keys)
        proof = {
            "cursor": [cp[k] for k in ("step", "epoch", "next_batch")],
            "optimizer_steps": sorted({int(v["step"]) for v in cp["optimizer"]["state"].values()}),
            "finite": all(torch.isfinite(t).all().item() for t in state.values()),
            "bn_buffers_unchanged": final_bn == initial_bn,
            "bn_affine_changed": fingerprint(state, affine) != initial_affine,
            "parameters_changed": fingerprint(state, set(state) - buffers) != initial_params,
            "bn_fingerprint": final_bn,
            "signature": cp["signature"],
            "timing": json.loads(timing.read_text()),
            "metrics": [json.loads(x) for x in (folder / "metrics.jsonl").read_text().splitlines()],
            "checkpoint": publish_arm(folder, experiment, arm),
        }
        if (
            proof["cursor"] != [90, 1, 0]
            or proof["optimizer_steps"] != [90]
            or not proof["finite"]
            or proof["bn_buffers_unchanged"] != (arm == "frozen")
            or not proof["bn_affine_changed"]
            or not proof["parameters_changed"]
        ):
            raise ValueError("Training acceptance failed")
        rng[arm] = (cp["rng_cpu"], cp["rng_cuda"])
        signatures[arm] = {k: v for k, v in cp["signature"].items() if k != "freeze_bn_stats"}
        arms[arm] = proof
        del cp, state
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Progress: training arm {arm} complete", flush=True)
    matched_rng = torch.equal(rng["normal"][0], rng["frozen"][0]) and all(
        torch.equal(a, b) for a, b in zip(rng["normal"][1], rng["frozen"][1], strict=True)
    )
    if not matched_rng or signatures["normal"] != signatures["frozen"]:
        raise ValueError("Training input signatures or RNG consumption differ")
    models = dict(
        zip(
            VARIANTS,
            (
                load_model(p, "cuda")
                for p in (initial, out / "normal/last.pt", out / "frozen/last.pt")
            ),
        )
    )
    before = {v: fingerprint(m.state_dict(), m.state_dict()) for v, m in models.items()}
    provenance_json = (DATA / "provenance.json").read_text()
    pinned = json.loads((ROOT / "docs/results/musdb18-training-provenance.json").read_text())
    expected = validation_inventory(provenance_json, dataset["provenance_sha256"], pinned)
    results = out / "evaluation"
    results.mkdir()
    records, listening = [], []
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    with threadpool_limits(limits=2):
        for index, track_id in enumerate(sorted(expected)):
            tick = time.monotonic()
            targets = {}
            for stem in ("mixture", *STEMS):
                path = DATA / "train-pool" / track_id.removeprefix("train/") / (stem + ".wav")
                if digest(path) != expected[track_id]["files"][stem]["sha256"]:
                    raise ValueError("Evaluation input changed")
                audio, rate = sf.read(path, dtype="float32", always_2d=True)
                if rate != RATE or audio.shape != (expected[track_id]["samples"], 2):
                    raise ValueError("Unexpected full-song audio shape")
                targets[stem] = audio.T.copy()
            waveform = torch.from_numpy(targets["mixture"])
            variants = {
                v: {s: x.numpy() for s, x in separate(m, waveform).items()}
                for v, m in models.items()
            }
            scored = score_variants(
                targets, variants, STEMS, length_policy="strict", variant_names=VARIANTS
            )
            record = {
                "id": track_id,
                "input_sha256": {s: x["sha256"] for s, x in expected[track_id]["files"].items()},
                "decoded_samples": waveform.shape[1],
                "segment_start_sample": 0,
                "seconds": time.monotonic() - tick,
                **scored,
            }
            records.append(record)
            atomic_json(results / f"track-{index:02d}.json", record)
            if index < 2:
                count = RATE * 8
                start = (waveform.shape[1] - count) // 2
                clips = {
                    f"sample-{index + 1}/{group}-{s}.wav": a[:, start : start + count]
                    for group, stems in {"reference": targets, **variants}.items()
                    for s, a in stems.items()
                }
                peak = max(float(np.abs(a).max()) for a in clips.values())
                gain = min(1.0, 0.95 / max(peak, 1e-12))
                for name, audio in clips.items():
                    path = results / name
                    path.parent.mkdir(exist_ok=True)
                    sf.write(path, audio.T * gain, RATE, subtype="FLOAT")
                listening.append(
                    {
                        "id": track_id,
                        "samples": count,
                        "start_sample": start,
                        "gain": gain,
                        "files": sorted(clips),
                    }
                )
            print(f"Progress: evaluated validation songs {index + 1}/10", flush=True)
    unchanged = all(
        before[v] == fingerprint(m.state_dict(), m.state_dict()) for v, m in models.items()
    )
    report = {
        "kind": "frozen-bn-pilot",
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
        "arms": arms,
        "matched_rng": matched_rng,
        "weights_unchanged": unchanged,
        "provenance_json": provenance_json,
        "records": records,
        "summary": summarize(records, expected),
        "listening": listening,
        "seconds": time.monotonic() - started,
        "evaluation_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    (results / "LISTENING.txt").write_text(
        "MUSDB18 research evaluation excerpts; original source licenses apply.\n"
        "Source: https://zenodo.org/records/1117372 and https://sigsep.github.io/datasets/musdb.html\n"
        "Artist/title and interval in report.json/listening. Educational use only.\n"
        "First two sorted validation songs, centered 8 seconds, one shared gain per song.\n"
        "reference-mixture: input; reference-*: targets; official_converted-*: baseline;\n"
        "normal_step90-*: ordinary BN; frozen_step90-*: fixed BN statistics.\n"
        "Metrics use unscaled full recordings. No subjective listening verdict claimed.\n"
    )
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
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.write_text(json.dumps(report, indent=2))
    from frozen_result import validate_frozen

    validate_frozen(
        {"status": "passed", "report": report},
        os.environ["SPT_COMMIT"],
        os.environ["RUNPOD_POD_ID"],
        report["storage"],
        os.environ["SPT_EXPERIMENT"],
    )
