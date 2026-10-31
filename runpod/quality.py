"""Pinned, five-preview 2/4-stem quality pilot. Downloads only public research assets."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = json.loads((ROOT / "runpod/quality-assets.json").read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(name, folder):
    spec = ASSETS["assets"][name]
    path = folder / name
    req = urllib.request.Request(spec["url"], headers={"User-Agent": "spleeterpt-quality/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response, path.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
        raise ValueError("Research asset checksum mismatch: " + name)
    return path


def command(*args):
    print("Running: " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, cwd=ROOT)


def run():
    import torch

    if torch.__version__ != "2.8.0+cu128" or not torch.cuda.is_available():
        raise RuntimeError("Quality pilot requires stable torch==2.8.0+cu128 and CUDA")
    folder = Path("/workspace/quality-assets")
    folder.mkdir(exist_ok=False)
    data = folder / "data/test"
    data.mkdir(parents=True)
    archive = download("preview.zip", folder)
    with zipfile.ZipFile(archive) as z:
        for name, expected in ASSETS["tracks"].items():
            path = data / name
            with z.open("test/" + name) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target)
            if digest(path) != expected:
                raise ValueError("Preview differs from pinned prior experiment: " + name)
    for name, repo in (("upstream", "deezer/spleeter"), ("legacy", "fwcd/spleeter-pytorch")):
        command("git", "clone", "https://github.com/" + repo + ".git", folder / name)
        command("git", "-C", folder / name, "checkout", "--detach", ASSETS[name + "_commit"])
    output = Path("/workspace/quality-results")
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    for stems in (2, 4):
        weights = folder / f"{stems}stems"
        weights.mkdir()
        archive = download(f"{stems}stems.tar.gz", folder)
        with tarfile.open(archive) as tar:
            tar.extractall(weights, filter="data")
        checkpoint = folder / f"{stems}stems.pt"
        command(
            sys.executable,
            "-m",
            "spleeterpt.cli",
            "convert",
            "--checkpoint",
            weights / "model",
            "--stems",
            stems,
            "--output",
            checkpoint,
        )
        command(
            sys.executable,
            "scripts/evaluate_musdb.py",
            "--root",
            data.parent,
            "--layout",
            "stems",
            "--subsets",
            "test",
            "--expected-tracks",
            5,
            "--dataset-label",
            "MUSDB18-7-STEMS five pinned preview clips",
            "--upstream-root",
            folder / "upstream",
            "--legacy-root",
            folder / "legacy",
            "--tf-checkpoint",
            weights / "model",
            "--checkpoint",
            checkpoint,
            "--device",
            "cuda",
            "--output",
            output / f"{stems}stems",
        )
        command(
            sys.executable,
            "scripts/evaluate_musdb.py",
            "--validate-output",
            output / f"{stems}stems",
        )
    return {
        "kind": "quality-preview",
        "commit": os.environ["SPT_COMMIT"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "seconds": time.monotonic() - started,
        "files": {
            str(p.relative_to(output)): p.read_text() for p in sorted(output.rglob("*.json"))
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(), indent=2))
