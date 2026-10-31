"""Full-size 2-stem CUDA acceptance on synthetic audio; not a quality benchmark."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from spleeterpt.checkpoint import read_checkpoint
from spleeterpt.cli import main
from spleeterpt.config import ModelConfig


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run():
    assert torch.cuda.is_available(), "CUDA is unavailable"
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    config = ModelConfig.preset(2)
    with tempfile.TemporaryDirectory(dir="/workspace", prefix="spt-smoke-") as temporary:
        root = Path(temporary)
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config.to_dict()))
        for split, seed in (("train", 0), ("validation", 1)):
            track = root / split
            track.mkdir()
            sources = (
                np.random.default_rng(seed)
                .normal(0, 0.01, (2, config.segment_samples * 2, 2))
                .astype("float32")
            )
            for name, samples in zip(config.instruments, sources):
                sf.write(track / f"{name}.wav", samples, 44100, subtype="FLOAT")
            sf.write(track / "mixture.wav", sources.sum(0), 44100, subtype="FLOAT")
            (root / f"{split}.jsonl").write_text(json.dumps({"track": str(track)}) + "\n")
        output = root / "run"
        common = [
            "train",
            "--config",
            str(config_path),
            "--train-manifest",
            str(root / "train.jsonl"),
            "--val-manifest",
            str(root / "validation.jsonl"),
            "--output",
            str(output),
            "--device",
            "cuda",
            "--precision",
            "fp32",
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--crops-per-track",
            "3",
            "--validation-crops",
            "1",
            "--save-every",
            "1",
            "--epochs",
            "1",
        ]
        main(common + ["--max-steps", "2"])
        first = read_checkpoint(output / "last.pt")
        assert first["step"] == 2 and first["next_batch"] == 2
        assert first["optimizer"]["state"], "Optimizer state missing"
        main(common + ["--resume", str(output / "last.pt")])
        last = read_checkpoint(output / "last.pt")
        assert last["step"] == 3 and last["epoch"] == 1
        assert any(not torch.equal(first["model"][k], v) for k, v in last["model"].items())
        assert all(torch.isfinite(v).all() for v in last["model"].values())
        records = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        train = [r for r in records if r["kind"] == "train"]
        validation = [r for r in records if r["kind"] == "validation"]
        assert [r["step"] for r in train] == [1, 2, 3]
        assert len(validation) == 1
        assert all(math.isfinite(r["loss"]) for r in records)
        assert all(not r["update_skipped"] and math.isfinite(r["grad_norm"]) for r in train)
        main(
            [
                "separate",
                str(root / "validation/mixture.wav"),
                "--checkpoint",
                str(output / "best.pt"),
                "--output",
                str(root / "stems"),
                "--device",
                "cuda",
            ]
        )
        for name in config.instruments:
            audio, rate = sf.read(root / "stems" / f"{name}.wav")
            assert audio.shape == (config.segment_samples * 2, 2) and rate == 44100
            assert np.isfinite(audio).all()
        torch.cuda.synchronize()
        return {
            "commit": os.environ["SPT_COMMIT"],
            "stems": 2,
            "precision": "fp32",
            "steps": [first["step"], last["step"]],
            "train_steps": [r["step"] for r in train],
            "validation_count": len(validation),
            "validation_loss": validation[0]["loss"],
            "optimizer_present": True,
            "weights_changed": True,
            "inference_finite": True,
            "last_sha256": sha256(output / "last.pt"),
            "best_sha256": sha256(output / "best.pt"),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "seconds": time.monotonic() - start,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "metrics": records,
            "packages": subprocess.check_output(["python", "-m", "pip", "freeze"], text=True),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(run(), indent=2))
