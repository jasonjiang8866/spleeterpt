"""Read-only paired full-song evaluation of the pinned persistent epoch-1 pilot."""

import argparse
import fcntl
import importlib.metadata
import json
import os
import re
import sys
import tarfile
import time
from pathlib import Path

from persistent import DATA, ROOT, verify_data
from quality import command, digest, download

sys.path.insert(0, str(ROOT / "scripts"))
from finetune_evaluation import (
    PROTOCOL,
    RATE,
    STEMS,
    VARIANTS,
    summarize_pair,
    validate_records,
    validation_inventory,
)
from musdb_evaluation import atomic_json, score_variants


def run():
    import numpy as np
    import soundfile as sf
    import torch
    from threadpoolctl import threadpool_limits

    from spleeterpt.audio import separate
    from spleeterpt.backup import validate
    from spleeterpt.checkpoint import load_model, read_checkpoint
    from spleeterpt.config import ModelConfig

    if not os.path.ismount("/data") or os.environ["SPT_EXPERIMENT"] != "41189465452-1":
        raise RuntimeError("Expected mounted persistent pilot")
    if torch.__version__ != "2.8.0+cu128" or not torch.cuda.is_available():
        raise RuntimeError("Expected stable CUDA runtime")
    if importlib.metadata.version("museval") != "0.4.1":
        raise RuntimeError("Expected pinned BSSEval implementation")
    torch.set_num_threads(2)
    torch.manual_seed(17)
    np.random.seed(17)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    evidence = json.loads((ROOT / "docs/results/persistent-pilot-41189465452.json").read_text())
    pinned = json.loads((ROOT / "docs/results/musdb18-training-provenance.json").read_text())
    with Path("/data/.spleeterpt.lock").open("a") as lock, threadpool_limits(limits=2):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dataset = verify_data()
        if dataset != evidence["resume"]["dataset"]:
            raise RuntimeError("Persistent dataset changed")
        provenance_json = (DATA / "provenance.json").read_text()
        expected = validation_inventory(provenance_json, dataset["provenance_sha256"], pinned)
        backup_root = Path("/data/experiments/41189465452-1/backups")
        folder, backup = validate(backup_root)
        if backup != evidence["resume"]["backup"]:
            raise RuntimeError("Expected exact verified final step-90 generation")
        checkpoint = folder / "last.pt"
        cp = read_checkpoint(checkpoint)
        if [cp[k] for k in ("step", "epoch", "next_batch")] != [90, 1, 0]:
            raise RuntimeError("Unexpected candidate training cursor")
        del cp
        pod_id = os.environ["RUNPOD_POD_ID"]
        if not re.fullmatch(r"[a-zA-Z0-9]+", pod_id):
            raise RuntimeError("Invalid evaluation identity")
        out = Path("/data/evaluations/41189465452-1") / pod_id
        out.mkdir(parents=True)
        assets = Path("/workspace/finetune-assets")
        assets.mkdir()
        archive = download("4stems.tar.gz", assets)
        with tarfile.open(archive) as tar:
            tar.extractall(assets / "weights", filter="data")
        initial = assets / "initial.pt"
        command(
            sys.executable,
            "-m",
            "spleeterpt.cli",
            "convert",
            "--checkpoint",
            assets / "weights/model",
            "--stems",
            "4",
            "--output",
            initial,
        )
        # Reconstructed bytes must match the actual initialization used in training.
        if digest(initial) != evidence["resume"]["initial_sha256"]:
            raise RuntimeError("Official initialization differs from training baseline")
        models = dict(zip(VARIANTS, [load_model(initial, "cuda"), load_model(checkpoint, "cuda")]))
        if any(m.config != ModelConfig.preset(4) for m in models.values()):
            raise RuntimeError("Expected identical full four-stem configurations")
        before = {
            v: {k: t.detach().cpu().clone() for k, t in m.state_dict().items()}
            for v, m in models.items()
        }
        if all(torch.equal(before[VARIANTS[0]][k], t) for k, t in before[VARIANTS[1]].items()):
            raise RuntimeError("Candidate is unchanged from initialization")
        records, listening = [], []
        for index, track_id in enumerate(sorted(expected)):
            tick = time.monotonic()
            track = DATA / "train-pool" / track_id.removeprefix("train/")
            targets = {}
            for stem in ("mixture", *STEMS):
                path = track / f"{stem}.wav"
                if digest(path) != expected[track_id]["files"][stem]["sha256"]:
                    raise RuntimeError("Input changed after data verification")
                audio, rate = sf.read(path, dtype="float32", always_2d=True)
                if rate != RATE or audio.shape != (expected[track_id]["samples"], 2):
                    raise RuntimeError("Unexpected full-track audio shape")
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
            atomic_json(out / f"track-{index:02d}.json", record)
            if index < 2:
                count = RATE * 8
                start = (waveform.shape[1] - count) // 2
                groups = {"reference": targets, **variants}
                clips = {
                    f"sample-{index + 1}/{group}-{stem}.wav": a[:, start : start + count]
                    for group, stems in groups.items()
                    for stem, a in stems.items()
                }
                peak = max(float(np.abs(a).max()) for a in clips.values())
                gain = min(1.0, 0.95 / max(peak, 1e-12))
                for name, a in clips.items():
                    path = out / name
                    path.parent.mkdir(exist_ok=True)
                    sf.write(path, a.T * gain, RATE, subtype="FLOAT")
                listening.append(
                    {
                        "id": track_id,
                        "start_sample": start,
                        "samples": count,
                        "gain": gain,
                        "files": sorted(clips),
                    }
                )
            print(f"Progress: evaluated validation songs {index + 1}/10", flush=True)
        validate_records(records, expected)
        unchanged = all(
            torch.equal(before[v][k], t.cpu())
            for v, m in models.items()
            for k, t in m.state_dict().items()
        )
        if not unchanged or validate(backup_root)[1] != backup:
            raise RuntimeError("Evaluation modified weights or persistent generation")
        (out / "LISTENING.txt").write_text(
            "MUSDB18 research evaluation excerpts, not repository-licensed audio.\n"
            "Source: https://zenodo.org/records/1117372 (MUSDB18; educational use only).\n"
            "Artist and title are recorded in report.json/listening; original licensing applies.\n"
            "See https://sigsep.github.io/datasets/musdb.html for rights and source attribution.\n"
            "Source-specific terms apply (Mixing Secrets / MedleyDB / other contributors).\n"
            "First two sorted validation songs; centered eight seconds, full-song inference.\n"
            "reference-mixture: input; reference-*: ground truth; official_converted-*: baseline;\n"
            "finetuned_step90-*: candidate. All files within a song share a single attenuation.\n"
            "Scores use original unscaled complete recordings, not these excerpts.\n"
            "No subjective listening verdict has been claimed by this automated run.\n"
        )
        report = {
            "kind": "finetune-validation",
            "commit": os.environ["SPT_COMMIT"],
            "pod_id": os.environ["RUNPOD_POD_ID"],
            "volume_id": os.environ["SPT_VOLUME_ID"],
            "experiment": os.environ["SPT_EXPERIMENT"],
            "results_directory": str(out),
            "protocol": PROTOCOL,
            "dataset": dataset,
            "provenance_json": provenance_json,
            "backup": backup,
            "initial_sha256": digest(initial),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "packages": {
                n: importlib.metadata.version(n)
                for n in ("museval", "numpy", "scipy", "soundfile", "tensorflow-cpu")
            },
            "weights_unchanged": unchanged,
            "records": records,
            "summary": summarize_pair(records),
            "listening": listening,
            "seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
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
        report["bundle"] = {
            "bytes": bundle.stat().st_size,
            "sha256": digest(bundle),
            "files": files,
        }
        return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = run()
    a.output.write_text(json.dumps(r, indent=2, allow_nan=False))
    from finetune_result import validate_finetune

    validate_finetune(
        {"status": "passed", "report": r},
        os.environ["SPT_COMMIT"],
        os.environ["SPT_VOLUME_ID"],
        os.environ["RUNPOD_POD_ID"],
    )
