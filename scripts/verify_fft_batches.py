"""Frontend-only FFT shape matrix. Does not imply official-checkpoint acceptance."""

import argparse
import json
import math
import platform
import subprocess
from pathlib import Path

import torch
from torch.nn import functional as F

from spleeterpt import audio, streaming
from spleeterpt.config import ModelConfig


def sample_lengths():
    for windows in (0, 1, 5):
        for remainder in (1, 2, 7, 28, 31, 32, 255, 256, 511, 512):
            for offset in (-1, 0, 1):
                length = (windows * 512 + remainder) * 1024 - 4096 + offset
                if length > 0:
                    yield length


@torch.inference_mode()
def run(device, *, lengths=None):
    if lengths is None:
        lengths = sample_lengths()
    config = ModelConfig.preset(5)
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(773)
    report = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "scope": "frontend only; no model or waveform acceptance gate",
        "device": device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name()
        if device.startswith("cuda")
        else platform.processor(),
        "config": config.to_dict(),
        "seed": 773,
        "cases": [],
    }
    for length in lengths:
        waveform = torch.randn((2, length), generator=generator) * 0.1
        full = audio.stft(waveform.to(device), config, inference=True)
        frames = math.ceil((length + config.n_fft) / config.hop_length)
        padded = F.pad(waveform, (4096, 4096 + 1024))
        errors = {}
        for policy in ("native", "pad-long-tail"):
            maxima = []
            for start in range(0, frames, 512):
                count = min(512, frames - start)
                samples = padded[:, start * 1024 : start * 1024 + 4096 + (count - 1) * 1024]
                spec = streaming._block_stft(
                    samples.to(device),
                    config,
                    start,
                    count,
                    policy == "pad-long-tail",
                    analysis_stft=audio.stft,
                )
                maxima.append(float((spec - full[:, start : start + count]).abs().max()))
            errors[policy] = maxima
        report["cases"].append({"samples": length, "frames": frames, "errors": errors})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.device), indent=2) + "\n")
