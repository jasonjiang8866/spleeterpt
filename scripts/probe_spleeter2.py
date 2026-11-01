"""Recreate and fingerprint the native two-stem official baseline on CPU."""

import argparse
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runpod"))
from quality import ASSETS, command, digest, download


def run(folder, output):
    import torch

    from spleeterpt.audio import separate
    from spleeterpt.checkpoint import load_model
    from spleeterpt.config import ModelConfig

    if torch.__version__ != "2.8.0+cpu" or torch.cuda.is_available():
        raise ValueError("Isolated Torch 2.8 CPU required")
    torch.set_num_threads(2)
    folder.mkdir(parents=True, exist_ok=True)
    archive = download("2stems.tar.gz", folder)
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
        "2",
        "--output",
        initial,
    )
    model = load_model(initial, "cpu").eval()
    if model.config != ModelConfig.preset(2):
        raise ValueError("Incorrect baseline configuration")
    wave = torch.randn(2, 44100, generator=torch.Generator().manual_seed(17)) * 0.03
    with torch.inference_mode():
        predictions = separate(model, wave)
    if set(predictions) != {"vocals", "accompaniment"} or any(
        x.shape != wave.shape or not torch.isfinite(x).all() for x in predictions.values()
    ):
        raise ValueError("Invalid Spleeter-2 smoke output")
    report = {
        "status": "passed",
        "kind": "spleeter2-cpu-probe",
        "torch": torch.__version__,
        "official_archive": ASSETS["assets"]["2stems.tar.gz"],
        "converted_sha256": digest(initial),
        "converted_bytes": initial.stat().st_size,
        "music_used": False,
        "gpu_used": False,
        "test_pool_used": False,
        "output_samples": 44100,
        "stems": sorted(predictions),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path("/tmp/spleeter2-assets"))
    parser.add_argument("--output", type=Path, default=Path("runpod-results/spleeter2-probe.json"))
    args = parser.parse_args()
    run(args.assets, args.output)
