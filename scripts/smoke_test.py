"""Exercise train -> resume -> separate using temporary synthetic audio.

No dataset download, pretrained weights, or cloud provisioning. Not a quality test.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from spleeterpt.checkpoint import read_checkpoint
from spleeterpt.cli import main
from spleeterpt.config import ModelConfig


def run(device, precision):
    torch.set_num_threads(2)
    config = ModelConfig(
        filters=(2, 3, 4, 5, 6, 7),
        sample_rate=8000,
        n_fft=128,
        hop_length=32,
        segment_frames=64,
        frequency_bins=64,
    )
    with tempfile.TemporaryDirectory(prefix="spleeterpt-smoke-") as temporary:
        root = Path(temporary)
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config.to_dict()))
        for split in ("train", "validation"):
            track = root / split
            track.mkdir()
            rng = np.random.default_rng(0 if split == "train" else 1)
            sources = rng.normal(0, 0.05, (2, config.segment_samples * 2, 2)).astype("float32")
            for name, audio in zip(config.instruments, sources):
                sf.write(track / f"{name}.wav", audio, config.sample_rate, subtype="FLOAT")
            sf.write(track / "mixture.wav", sources.sum(0), config.sample_rate, subtype="FLOAT")
            (root / f"{split}.jsonl").write_text(json.dumps({"track": str(track)}) + "\n")
        common = [
            "train",
            "--config",
            str(config_path),
            "--train-manifest",
            str(root / "train.jsonl"),
            "--val-manifest",
            str(root / "validation.jsonl"),
            "--output",
            str(root / "run"),
            "--device",
            device,
            "--precision",
            precision,
            "--batch-size",
            "1",
            "--workers",
            "0",
            "--crops-per-track",
            "2",
            "--validation-crops",
            "1",
            "--save-every",
            "1",
        ]
        main(common + ["--epochs", "1"])
        main(common + ["--epochs", "2", "--resume", str(root / "run" / "last.pt")])
        assert read_checkpoint(root / "run" / "last.pt")["step"] == 4
        main(
            [
                "separate",
                str(root / "validation" / "mixture.wav"),
                "--checkpoint",
                str(root / "run" / "best.pt"),
                "--output",
                str(root / "stems"),
                "--device",
                device,
            ]
        )
        for name in config.instruments:
            audio, sample_rate = sf.read(root / "stems" / f"{name}.wav")
            assert audio.shape == (config.segment_samples * 2, 2)
            assert sample_rate == config.sample_rate and np.isfinite(audio).all()
    print(
        f"PASS: synthetic training, resume, checkpoint reload, stereo inference ({device}/{precision})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    args = parser.parse_args()
    run(args.device, args.precision)
