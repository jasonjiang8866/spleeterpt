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


def run(output):
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
    validate_runtime(runtime, MIG24)
    work = Path("/workspace/bounded")
    work.mkdir()
    export = work / "export"
    export.mkdir()
    report = {
        "kind": "bounded-memory-v1",
        "commit": os.environ["SPT_COMMIT"],
        "pod_id": os.environ["RUNPOD_POD_ID"],
        "experiment": os.environ["SPT_EXPERIMENT"],
        "global_volume_id": os.environ["SPT_GLOBAL_VOLUME_ID"],
        "gpu_runtime": runtime,
        "benchmarks": {},
        "complete": False,
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
        for stems in (2, 4):
            checkpoint = download_model(f"{stems}stems", work / "weights")
            for kind, durations in (("synthetic", [1, 12, 60, 600]), ("real", [1, 12, 60])):
                key = f"{stems}stems-{kind}"
                destination = export / f"{key}.json"
                try:
                    command(
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
                    )
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
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
