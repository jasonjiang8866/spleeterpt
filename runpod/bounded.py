"""One MIG experiment: released weights, isolated synthetic/real file comparisons."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from global_inputs import PREFIX, checked_path, copy_file, encoded, manifest
from gpu_profile import MIG24, validate_runtime
from quality import ROOT, command, digest


def measured_command(destination, *args, collect_rejection=False):
    """Continue only for a complete numerical rejection, never a crashed worker."""
    try:
        command(*args)
        return True
    except subprocess.CalledProcessError as error:
        if not collect_rejection or error.returncode != 1 or not destination.is_file():
            raise
        report = json.loads(destination.read_text())
        start = args.index("--durations") + 1
        end = next(i for i in range(start, len(args)) if str(args[i]).startswith("--"))
        durations = [float(x) for x in args[start:end]]
        if (
            report.get("numerical_pass") is not False
            or [c.get("seconds") for c in report.get("cases", [])] != durations
            or not all("stems" in c and "measurements" in c for c in report["cases"])
        ):
            raise
        print("Recorded numerical rejection; collecting remaining evidence", flush=True)
        return False


def run(output, *, five_stem=False):
    import torch

    from spleeterpt.pretrained import download_model

    if torch.__version__ != "2.8.0+cu128" or not torch.cuda.is_available():
        raise RuntimeError("Requires pinned torch 2.8.0+cu128 and real CUDA")
    runtime = {
        "requested_gpu": os.environ["SPT_GPU_ID"],
        "name": torch.cuda.get_device_name(),
        "device_count": torch.cuda.device_count(),
        "capability": list(torch.cuda.get_device_capability()),
        "total_memory_bytes": torch.cuda.get_device_properties().total_memory,
    }
    validate_runtime(runtime, runtime["requested_gpu"] if five_stem else MIG24)
    work = Path("/workspace/bounded")
    work.mkdir()
    export = work / "export"
    export.mkdir()
    report = {
        "kind": "five-stem-bounded-v1" if five_stem else "bounded-memory-v1",
        "commit": os.environ["SPT_COMMIT"],
        "pod_id": os.environ["RUNPOD_POD_ID"],
        "experiment": os.environ["SPT_EXPERIMENT"],
        "global_volume_id": os.environ["SPT_GLOBAL_VOLUME_ID"],
        "gpu_runtime": runtime,
        "benchmarks": {},
        "complete": False,
        "numerical_rejections": [],
    }

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        if not os.path.ismount("/global"):
            raise ValueError("Existing Global Volume required")
        source = Path("/global") / PREFIX
        provenance = checked_path(source, "musdb18/provenance.json").read_text()
        snapshot = manifest(provenance)
        raw = checked_path(source, "snapshot.json").read_bytes()
        if raw != encoded(snapshot):
            raise ValueError("Pinned validation snapshot mismatch")
        # Read/copy only one previously evaluated validation mixture; no test pool.
        name = min(n for n in snapshot["files"] if n.endswith("/mixture.wav"))
        real = work / "real.wav"
        copy_file(checked_path(source, name), real, snapshot["files"][name])
        report["real_input"] = {
            "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
            "path": name,
            **snapshot["files"][name],
            "test_pool_used": False,
        }
        save()
        tests = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_streaming.py", "-k", "cuda"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        (export / "cuda-tests.txt").write_text(tests.stdout + tests.stderr)
        report["cuda_tests"] = {"returncode": tests.returncode, "output": tests.stdout[-4000:]}
        if tests.returncode or "3 passed" not in tests.stdout:
            raise RuntimeError("CUDA 2/4/5-stem regression did not pass")
        if five_stem:
            command(sys.executable, "scripts/prepare_five_stem.py", "--work", work / "weights")
            report["conversion"] = json.loads((work / "weights/provenance.json").read_text())
            command(
                sys.executable,
                "scripts/verify_bounded_checkpoint.py",
                "--checkpoint",
                work / "weights/5stems.pt",
                "--device",
                "cuda",
                "--output",
                export / "edges.json",
            )
            report["edges"] = json.loads((export / "edges.json").read_text())
            save()
        for stems in (5,) if five_stem else (2, 4):
            checkpoint = (
                work / "weights/5stems.pt"
                if five_stem
                else download_model(f"{stems}stems", work / "weights")
            )
            if stems == 4:
                command(
                    sys.executable,
                    "scripts/diagnose_bounded.py",
                    "--checkpoint",
                    checkpoint,
                    "--input",
                    real,
                    "--device",
                    "cuda",
                    "--output",
                    export / "diagnostic.json",
                )
                report["diagnostic"] = json.loads((export / "diagnostic.json").read_text())
                save()
            for kind, durations in (("synthetic", [1, 12, 60, 600]), ("real", [1, 12, 60])):
                key = f"{stems}stems-{kind}"
                destination = export / f"{key}.json"
                try:
                    passed = measured_command(
                        destination,
                        sys.executable,
                        "scripts/benchmark_bounded.py",
                        "--checkpoint",
                        checkpoint,
                        "--device",
                        "cuda",
                        "--durations",
                        *durations,
                        "--output",
                        destination,
                        *(["--input", real] if kind == "real" else []),
                        collect_rejection=five_stem,
                    )
                    if not passed:
                        report["numerical_rejections"].append(key)
                finally:
                    if destination.exists():
                        report["benchmarks"][key] = json.loads(destination.read_text())
                        save()
        if five_stem:
            command(
                sys.executable,
                "scripts/diagnose_bounded.py",
                "--checkpoint",
                checkpoint,
                "--input",
                real,
                "--device",
                "cuda",
                "--seconds",
                60,
                "--fp64-control",
                "--output",
                export / "diagnostic.json",
            )
            report["diagnostic"] = json.loads((export / "diagnostic.json").read_text())
            save()
            import soundfile as sf

            from spleeterpt.audio import read_audio
            from spleeterpt.pretrained import verified_download

            piano = json.loads((ROOT / "scripts/piano_source.json").read_text())
            original = verified_download(
                piano["url"], work / "piano.ogg", sha256=piano["sha256"], size=piano["bytes"]
            )
            audio = read_audio(original, 44100)
            report["piano_source"] = piano
            report["piano_inputs"] = {}
            for start in piano["starts_seconds"]:
                clip = work / f"piano-{start}.wav"
                waveform = audio[:, start * 44100 : (start + 12) * 44100]
                if waveform.shape != (2, 12 * 44100):
                    raise ValueError("Incomplete piano clip")
                sf.write(clip, waveform.numpy().T, 44100, subtype="FLOAT")
                key = f"5stems-piano-{start}"
                report["piano_inputs"][key] = {
                    "sha256": digest(clip),
                    "start_seconds": start,
                    "samples": 12 * 44100,
                }
                destination = export / f"{key}.json"
                try:
                    passed = measured_command(
                        destination,
                        sys.executable,
                        "scripts/benchmark_bounded.py",
                        "--checkpoint",
                        checkpoint,
                        "--device",
                        "cuda",
                        "--durations",
                        12,
                        "--input",
                        clip,
                        "--output",
                        destination,
                        collect_rejection=True,
                    )
                    if not passed:
                        report["numerical_rejections"].append(key)
                finally:
                    if destination.exists():
                        report["benchmarks"][key] = json.loads(destination.read_text())
                    save()
        report["complete"] = True
    finally:
        (export / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        files = {
            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(export.iterdir())
            if p.is_file()
        }
        bundle = Path("/workspace/checkpoint.tar")
        with tarfile.open(bundle, "w") as tar:
            for name in files:
                tar.add(export / name, arcname=name, recursive=False)
        report["bundle"] = {
            "bytes": bundle.stat().st_size,
            "sha256": digest(bundle),
            "files": files,
        }
        save()
    if report["numerical_rejections"]:
        raise RuntimeError("Numerical acceptance failed; full evidence retained")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(
        parser.parse_args().output, five_stem=os.environ.get("SPT_BENCHMARK_PROFILE") == "five-stem"
    )
