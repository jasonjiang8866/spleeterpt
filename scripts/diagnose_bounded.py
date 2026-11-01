"""Short, explicitly unbounded diagnostic isolating FFT batch-size roundoff."""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from spleeterpt import streaming
from spleeterpt.audio import separate, stft
from spleeterpt.checkpoint import load_model


@torch.inference_mode()
def diagnose(checkpoint, input_path, device):
    model = load_model(checkpoint, device)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    config = model.config
    samples, rate = sf.read(
        input_path, frames=12 * config.sample_rate, dtype="float32", always_2d=True
    )
    if rate != config.sample_rate or samples.shape != (12 * rate, 2):
        raise ValueError("Diagnostic requires at least 12 seconds of native-rate stereo")
    waveform = torch.from_numpy(samples.T.copy())
    reference = separate(model, waveform)

    def collect():
        blocks = list(streaming.separate_iter(model, waveform))
        return {n: torch.cat([b[n] for b in blocks], -1) for n in config.instruments}

    normal = collect()
    full_spectrum = stft(waveform.to(device), config, inference=True)
    original_stft = streaming.stft
    frame_start, fft_errors = 0, []

    def cached_spectrum(audio, config):
        nonlocal frame_start
        actual = original_stft(audio, config)
        expected = full_spectrum[:, frame_start : frame_start + actual.shape[-2]]
        fft_errors.append(float((actual - expected).abs().max()))
        frame_start += actual.shape[-2]
        return expected

    try:
        streaming.stft = cached_spectrum
        cached = collect()
    finally:
        streaming.stft = original_stft

    def metrics(other):
        return {
            name: {
                "max_abs_error": float((other[name] - ref).abs().max()),
                "rms_error": float((other[name].double() - ref.double()).square().mean().sqrt()),
                "reference_rms": float(ref.double().square().mean().sqrt()),
            }
            for name, ref in reference.items()
        }

    return {
        "duration_seconds": 12,
        "device": device,
        "purpose": "normal vs cached full-STFT blocks; cached path isolates synthesis grouping",
        "stft_block_max_complex_errors": fft_errors,
        "normal_bounded": metrics(normal),
        "cached_full_stft_bounded": metrics(cached),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(diagnose(args.checkpoint, args.input, args.device), indent=2) + "\n"
    )
