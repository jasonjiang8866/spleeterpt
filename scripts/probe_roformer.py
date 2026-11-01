"""Real-weight CPU compatibility and original-author parity; no music/GPU usage."""

import argparse
import gc
import importlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from roformer_kim import (
    MODEL_CONFIG,
    RECIPE,
    SOURCE_FILES,
    WEIGHT_SHA256,
    checked_download,
    digest,
    load_model,
    peak_rss_bytes,
    recipe_sha256,
    separate,
)

AUTHOR_COMMIT = "25f44ffb55ee3c301281bba21b2d6d311cb69ae2"
AUTHOR_FILES = {
    "models/mel_band_roformer/__init__.py": (
        71,
        "bde4b0739ca79d0d85ee9cf1df6eea0558c8e17e06414fbb3600897ff6d24640",
    ),
    "models/mel_band_roformer/attend.py": (
        2854,
        "9ffd7d06b22a53bb8e6162adf674e7298d2227e1eea4f6404cb89ff1bbefecbe",
    ),
    "models/mel_band_roformer/mel_band_roformer.py": (
        16726,
        "d0e958c58114872c3a0af54f97719e07690ff5d98473fcd8b48485f776737b8f",
    ),
    "utils.py": (4161, "c68f49bb593347e256f5001dea1f380e9c575c331afc6b2920e8cdbdcd188be2"),
    "configs/config_vocals_mel_band_roformer.yaml": (
        733,
        "87aabb300193b019159269b69d6fe5f313aae4201a22b4af6268bc2846ab2fe1",
    ),
}


def signal(torch):
    t = torch.arange(44100, dtype=torch.float32) / 44100
    carrier = sum(
        torch.sin(2 * torch.pi * f * t) / (i + 1) for i, f in enumerate((220, 440, 660, 880))
    )
    wave = carrier * (0.08 + 0.04 * torch.sin(2 * torch.pi * 3 * t))
    return torch.stack((wave, wave.roll(31) * 0.8))


def original(args):
    import numpy as np
    import torch
    import yaml

    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    folder = args.assets / "original"
    for name, (size, sha) in AUTHOR_FILES.items():
        checked_download(
            f"https://raw.githubusercontent.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/{AUTHOR_COMMIT}/{name}",
            folder / name,
            size,
            sha,
        )
    sys.path.insert(0, str(folder.resolve()))
    cls = importlib.import_module("models.mel_band_roformer").MelBandRoformer
    # FullLoader only for hash-authenticated original YAML's python/tuple scalar.
    config = yaml.load(
        (folder / "configs/config_vocals_mel_band_roformer.yaml").read_text(),
        Loader=yaml.FullLoader,
    )
    kwargs = config["model"]
    assert {k: MODEL_CONFIG[k] for k in kwargs if k != "flash_attn"} == {
        k: v for k, v in kwargs.items() if k != "flash_attn"
    }
    kwargs["flash_attn"] = False
    model = cls(**kwargs).float().eval()
    weight = args.assets / "MelBandRoformer.ckpt"
    if digest(weight) != WEIGHT_SHA256:
        raise ValueError("Original comparison weight mismatch")
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True), strict=True)
    with torch.inference_mode():
        output = model(signal(torch)[None])[0]
    np.save(args.output.with_suffix(".npy"), output.numpy())
    del model
    gc.collect()
    demix = importlib.import_module("utils").demix_track
    cfg = SimpleNamespace(
        inference=SimpleNamespace(chunk_size=100, num_overlap=2),
        training=SimpleNamespace(target_instrument="vocals", instruments=["vocals", "other"]),
    )

    class Scale(torch.nn.Module):
        def forward(self, x):
            return x * 0.375

    error = 0.0
    for length in (1, 49, 50, 51, 99, 100, 101, 199, 211):
        wave = torch.randn(2, length, generator=torch.Generator().manual_seed(17))
        reference, _ = demix(cfg, Scale(), wave, "cpu", first_chunk_time=1)
        actual = separate(Scale(), wave, chunk_size=100)["vocals"].numpy()
        if actual.shape != reference["vocals"].shape:
            raise ValueError("Author overlap-add length mismatch")
        error = max(error, float(np.max(np.abs(actual - reference["vocals"]))))
        np.testing.assert_allclose(actual, reference["vocals"], rtol=1e-6, atol=1e-6)
    args.output.with_suffix(".reference.json").write_text(
        json.dumps({"overlap_add_max_abs": error})
    )


def run(args):
    import numpy as np
    import torch

    started = time.monotonic()
    if torch.__version__ != "2.8.0+cpu" or torch.cuda.is_available():
        raise ValueError("CPU probe requires isolated torch 2.8.0+cpu")
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.use_deterministic_algorithms(True)
    model = load_model(args.assets)
    parameters = sum(p.numel() for p in model.parameters())
    wave = signal(torch)
    with torch.inference_mode():
        output = model(wave[None])[0].numpy()
    if output.shape != (2, 44100) or not np.isfinite(output).all():
        raise ValueError("Invalid CPU forward output")
    del model
    gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--assets",
            str(args.assets),
            "--output",
            str(args.output),
            "--original",
        ],
        check=True,
    )
    reference = np.load(args.output.with_suffix(".npy"))
    np.testing.assert_allclose(output, reference, rtol=1e-5, atol=1e-6)
    report = {
        "status": "passed",
        "kind": "roformer-cpu-probe",
        "music_used": False,
        "gpu_used": False,
        "test_pool_used": False,
        "recipe": RECIPE,
        "recipe_sha256": recipe_sha256(),
        "source_files": SOURCE_FILES,
        "author_commit": AUTHOR_COMMIT,
        "author_files": AUTHOR_FILES,
        "parameters": parameters,
        "torch": torch.__version__,
        "output_shape": list(output.shape),
        "output_peak": float(np.abs(output).max()),
        "author_forward_max_abs": float(np.max(np.abs(output - reference))),
        **json.loads(args.output.with_suffix(".reference.json").read_text()),
        "process_peak_rss_bytes": peak_rss_bytes(),
        "rss_source": "/proc/self/status VmHWM",
        "seconds": time.monotonic() - started,
        "packages": {
            d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()
        },
        "scope": "synthetic CPU compatibility and author FP32 eager parity; no GPU performance or music quality claim",
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "parameters",
                    "author_forward_max_abs",
                    "overlap_add_max_abs",
                    "seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path("/tmp/roformer-assets"))
    parser.add_argument("--output", type=Path, default=Path("runpod-results/roformer-probe.json"))
    parser.add_argument("--original", action="store_true")
    args = parser.parse_args()
    original(args) if args.original else run(args)
