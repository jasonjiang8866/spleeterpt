"""Rebuild and validate the pinned 2/4-stem release on CPU; never publish by itself."""

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from spleeterpt.pretrained import digest, verified_download

ROOT = Path(__file__).resolve().parents[1]
TAG = "weights-v1.0.0"
REPOSITORY = "jasonjiang8866/spleeterpt"
UPSTREAM = "c8854001ac8acad34a9bc2bd15f28475541828b1"
AUDIO_SHA256 = "4b431d535d235bd81b62f816f08c3f1afb6679d2706d84b3b75903b7df909507"
ENVIRONMENT = {"torch": "2.8.0+cpu", "tensorflow-cpu": "2.18.1", "numpy": "2.0.2"}


def unpack_checkpoint(archive, folder):
    """Extract only the three fixed TF files; no archive-provided paths or links."""
    folder.mkdir(parents=True, exist_ok=True)
    wanted = {"model.index", "model.meta", "model.data-00000-of-00001"}
    seen = set()
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.name not in wanted:
                continue
            if member.name in seen or not member.isfile() or not 0 < member.size < 500_000_000:
                raise ValueError("Unexpected checkpoint member")
            seen.add(member.name)
            with tar.extractfile(member) as source, (folder / member.name).open("wb") as target:
                shutil.copyfileobj(source, target)
    if seen != wanted:
        raise ValueError("Incomplete official checkpoint archive")


def command(arguments, log):
    print("Running " + " ".join(map(str, arguments)), flush=True)
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "2",
        "TF_NUM_INTEROP_THREADS": "2",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "CUDA_VISIBLE_DEVICES": "",
    }
    with log.open("w") as stream:
        subprocess.run(
            list(map(str, arguments)),
            check=True,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def build(work, output, upstream, expected_catalog=None):
    versions = {name: importlib.metadata.version(name) for name in ENVIRONMENT}
    if versions != ENVIRONMENT or sys.version_info[:2] != (3, 12):
        raise ValueError(f"Release environment mismatch: {versions}; need Python 3.12")
    ref = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if ref != UPSTREAM or digest(upstream / "audio_example.mp3") != AUDIO_SHA256:
        raise ValueError("Unpinned upstream checkout/example")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True):
        raise ValueError("Upstream checkout is modified")
    output.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=True)
    assets = json.loads((ROOT / "runpod/quality-assets.json").read_text())["assets"]
    catalog = {"schema_version": 1, "release": TAG, "models": {}}
    provenance = {
        "repository": REPOSITORY,
        "conversion_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_modified": bool(subprocess.check_output(["git", "diff", "HEAD"], text=True)),
        "upstream_code_commit": UPSTREAM,
        "environment": versions,
        "audio_fixture_sha256": AUDIO_SHA256,
        "license_basis": "https://doi.org/10.21105/joss.02154",
        "license_clarification_issue": "https://github.com/deezer/spleeter/issues/959",
        "validation": "original TF graph + upstream waveform pipeline, music and four edge cases",
        "training": "none; official pretrained tensor conversion only",
        "models": {},
    }
    for stems in (2, 4):
        name = f"{stems}stems"
        original = assets[name + ".tar.gz"]
        archive = verified_download(
            original["url"],
            work / (name + ".tar.gz"),
            sha256=original["sha256"],
            size=original["bytes"],
        )
        folder = work / name
        unpack_checkpoint(archive, folder)
        checkpoint = output / (name + ".pt")
        command(
            [
                sys.executable,
                "-m",
                "spleeterpt.cli",
                "convert",
                "--checkpoint",
                folder / "model",
                "--stems",
                stems,
                "--output",
                checkpoint,
            ],
            output / (name + "-conversion.log"),
        )
        item = {
            "filename": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
            "sha256": digest(checkpoint),
            "url": f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{checkpoint.name}",
            "upstream_archive": original,
            "license": "MIT",
            "stems": stems,
        }
        catalog["models"][name] = item
        if expected_catalog is not None and item != expected_catalog["models"][name]:
            raise ValueError(f"Rebuilt {name} differs from pinned catalog")
        command(
            [
                sys.executable,
                "scripts/verify_official_checkpoint.py",
                "--tf-checkpoint",
                folder / "model",
                "--checkpoint",
                checkpoint,
            ],
            output / (name + "-graph-parity.log"),
        )
        report_path = output / (name + "-waveform-parity.json")
        command(
            [
                sys.executable,
                "scripts/verify_audio_parity.py",
                upstream / "audio_example.mp3",
                "--tf-checkpoint",
                folder / "model",
                "--checkpoint",
                checkpoint,
                "--upstream-root",
                upstream,
                "--output",
                report_path,
                "--include-edges",
            ],
            output / (name + "-waveform-parity.log"),
        )
        parity = json.loads(report_path.read_text())
        if parity["passed"] is not True or len(parity["cases"]) != 5:
            raise ValueError("Incomplete waveform acceptance")
        from spleeterpt.checkpoint import read_checkpoint

        payload = read_checkpoint(checkpoint)
        (output / (name + "-config.json")).write_text(
            json.dumps(payload["config"], indent=2) + "\n"
        )
        provenance["models"][name] = {
            "checkpoint_sha256": item["sha256"],
            "source_tensor_files": payload["source_sha256"],
            "waveform_report_sha256": digest(report_path),
            "passed": True,
        }
        del payload
        print(f"PASS {name}: {item['bytes']} bytes SHA256={item['sha256']}", flush=True)
    if expected_catalog is not None and catalog != expected_catalog:
        raise ValueError("Release catalog metadata differs")
    (output / "pretrained.json").write_text(json.dumps(catalog, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    shutil.copyfile(ROOT / "licenses/Deezer-Spleeter-MIT.txt", output / "Deezer-Spleeter-MIT.txt")
    shutil.copyfile(ROOT / "docs/pretrained-release.md", output / "MODEL_CARD.md")
    (output / "SHA256SUMS").write_text(
        "".join(f"{digest(p)}  {p.name}\n" for p in sorted(output.iterdir()) if p.is_file())
    )
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument(
        "--catalog", type=Path, help="fail if rebuilt weights differ from this catalog"
    )
    args = parser.parse_args()
    expected = json.loads(args.catalog.read_text()) if args.catalog else None
    build(args.work.resolve(), args.output.resolve(), args.upstream_root.resolve(), expected)


if __name__ == "__main__":
    main()
