"""Short, explicitly unbounded diagnostic isolating FFT batch-size roundoff."""

import argparse
import json
import math
from pathlib import Path

import soundfile as sf
import torch

from spleeterpt import audio as audio_module
from spleeterpt import streaming
from spleeterpt.audio import separate, stft
from spleeterpt.checkpoint import load_model


@torch.inference_mode()
def diagnose(checkpoint, input_path, device, seconds=12, fp64_control=False):
    model = load_model(checkpoint, device)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    config = model.config
    samples, rate = sf.read(
        input_path, frames=seconds * config.sample_rate, dtype="float32", always_2d=True
    )
    if rate != config.sample_rate or samples.shape != (seconds * rate, 2):
        raise ValueError("Diagnostic requires the requested duration of native-rate stereo")
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

    def metrics(other, reference=reference):
        return {
            name: {
                "max_abs_error": float((other[name] - ref).abs().max()),
                "rms_error": float((other[name].double() - ref.double()).square().mean().sqrt()),
                "reference_rms": float(ref.double().square().mean().sqrt()),
                "samples_above_2e_6": int(((other[name] - ref).abs() > 2e-6).sum()),
                "samples_above_1e_6": int(((other[name] - ref).abs() > 1e-6).sum()),
                "scalar_samples": ref.numel(),
            }
            for name, ref in reference.items()
        }

    report = {
        "duration_seconds": seconds,
        "device": device,
        "purpose": "normal vs cached full-STFT blocks; cached path isolates synthesis grouping",
        "stft_block_max_complex_errors": fft_errors,
        "normal_bounded": metrics(normal),
        "cached_full_stft_bounded": metrics(cached),
    }
    if fp64_control:
        full64 = precise_stft(waveform.to(device), config, inference=True)
        errors64, frame_start = [], 0

        def block64(audio, config):
            nonlocal frame_start
            actual = precise_stft(audio, config)
            expected = full64[:, frame_start : frame_start + actual.shape[-2]]
            errors64.append(float((actual - expected).abs().max()))
            frame_start += actual.shape[-2]
            return actual

        try:
            audio_module.stft = precise_stft
            streaming.stft = block64
            reference64 = separate(model, waveform)
            bounded64 = collect()
        finally:
            audio_module.stft = original_stft
            streaming.stft = original_stft
        report["fp64_fft_control"] = {
            "scope": "FP64 FFT using the same FP32 Hann coefficients, cast to complex64 before unchanged FP32 network; diagnostic only",
            "stft_block_max_complex_errors": errors64,
            "bounded_vs_full": metrics(bounded64, reference64),
            "full_vs_original_fp32": metrics(reference64),
        }
    return report


def precise_stft(waveform, config, *, inference=False):
    """Independent FFT precision control; keep padding and FP32 window coefficients."""
    if inference:
        waveform = torch.nn.functional.pad(waveform, (config.n_fft, 0))
        frames = math.ceil(waveform.shape[-1] / config.hop_length)
        needed = config.n_fft + (frames - 1) * config.hop_length
        waveform = torch.nn.functional.pad(waveform, (0, needed - waveform.shape[-1]))
    window = torch.hann_window(config.n_fft, dtype=torch.float32, device=waveform.device)
    return (
        torch.stft(
            waveform.double(),
            config.n_fft,
            config.hop_length,
            window=window.double(),
            center=False,
            return_complex=True,
        )
        .transpose(-2, -1)
        .to(torch.complex64)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seconds", type=int, default=12)
    parser.add_argument("--fp64-control", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(
            diagnose(args.checkpoint, args.input, args.device, args.seconds, args.fp64_control),
            indent=2,
        )
        + "\n"
    )
