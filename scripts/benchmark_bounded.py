"""Isolated legacy/file-streaming memory, time and numerical comparison.

Every mode/length runs in a fresh process. CUDA measurements require --device
cuda and a real CUDA runtime; CPU results never stand in for GPU measurements.
SI-SDR uses legacy output as its reference, not source stems / separation quality.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from spleeterpt.audio import read_audio, separate
from spleeterpt.checkpoint import load_model, read_checkpoint
from spleeterpt.config import ModelConfig
from spleeterpt.streaming import separate_file


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_mib():
    # Linux ru_maxrss is KiB; macOS uses bytes. Includes interpreter/checkpoint
    # loading/warmup; unlike an allocator delta this is a real process watermark.
    unit = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / unit


def worker(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    model = load_model(args.checkpoint, args.device)
    cuda = next(model.parameters()).is_cuda
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # Same one-window warmup for both paths; excluded from elapsed seconds.
    for _ in range(args.warmup):
        separate(model, torch.zeros(2, model.config.hop_length))
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    before = peak_rss_mib()
    started = time.perf_counter()
    if args.worker == "legacy":
        output = Path(args.output)
        output.mkdir()
        audio = read_audio(args.input, model.config.sample_rate)
        for name, stem in separate(model, audio).items():
            sf.write(
                output / f"{name}.wav",
                stem.numpy().T,
                model.config.sample_rate,
                subtype="FLOAT",
                format="RF64" if stem.shape[-1] * 8 >= 2**32 - 4096 else "WAV",
            )
    else:
        separate_file(model, args.input, args.output)
    if cuda:
        torch.cuda.synchronize()
    result = {
        "mode": args.worker,
        "seconds_including_io": time.perf_counter() - started,
        "process_peak_rss_mib": peak_rss_mib(),
        "pre_inference_peak_rss_mib": before,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if cuda else None,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2 if cuda else None,
        "device": str(next(model.parameters()).device),
        "device_name": torch.cuda.get_device_name() if cuda else platform.processor(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_runtime": {
            "requested_gpu": os.environ.get("SPT_GPU_ID"),
            "name": torch.cuda.get_device_name(),
            "device_count": torch.cuda.device_count(),
            "capability": list(torch.cuda.get_device_capability()),
            "total_memory_bytes": torch.cuda.get_device_properties().total_memory,
        }
        if cuda
        else None,
    }
    print(json.dumps(result))


def paired_metrics(reference, estimate):
    """Global stereo, zero-mean SI-SDR in float64, with a stable second pass."""
    sums = np.zeros(5, dtype=np.float64)
    count, maximum, error2, reference_max = 0, 0.0, 0.0, 0.0
    with sf.SoundFile(reference) as ref, sf.SoundFile(estimate) as est:
        if (ref.frames, ref.channels, ref.samplerate) != (est.frames, est.channels, est.samplerate):
            raise ValueError("output shape/sample rate mismatch")
        while True:
            x, y = ref.read(65536, always_2d=True), est.read(65536, always_2d=True)
            if not len(x):
                break
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("non-finite output")
            count += x.size
            difference = x - y
            maximum = max(maximum, float(np.abs(difference).max()))
            reference_max = max(reference_max, float(np.abs(x).max()))
            error2 += float(np.square(difference).sum())
            sums += [x.sum(), y.sum(), np.square(x).sum(), np.square(y).sum(), (x * y).sum()]
        mx, my = sums[:2] / count
        xx = max(0.0, sums[2] - count * mx * mx)
        xy = sums[4] - count * mx * my
        alpha = xy / xx if xx > 0 else 0.0
        residual = 0.0
        ref.seek(0)
        est.seek(0)
        while True:
            x, y = ref.read(65536, always_2d=True), est.read(65536, always_2d=True)
            if not len(x):
                break
            residual += float(np.square(y - my - alpha * (x - mx)).sum())
    projection = alpha * alpha * xx
    score = 10 * math.log10(projection / residual) if projection > 0 and residual > 0 else None
    return {
        "samples_per_channel": count // 2,
        "max_abs_error": maximum,
        "rms_error": math.sqrt(error2 / count),
        "reference_peak": reference_max,
        "legacy_reference_si_sdr_db": score,
        "si_sdr_status": "finite"
        if score is not None
        else "undefined_silence_or_constant"
        if xx == 0
        else "exact_or_degenerate_projection",
    }


def make_input(path, config, seconds, input_path):
    length = round(seconds * config.sample_rate)
    if length <= 0:
        raise ValueError("durations must be positive")
    with sf.SoundFile(
        path,
        "w",
        samplerate=config.sample_rate,
        channels=2,
        subtype="FLOAT",
        format="RF64" if length * 8 >= 2**32 - 4096 else "WAV",
    ) as out:
        if input_path:
            with sf.SoundFile(input_path) as source:
                if source.samplerate != config.sample_rate or source.channels not in (1, 2):
                    raise ValueError("input must be mono/stereo at the model sample rate")
                if source.frames < length:
                    raise ValueError("input shorter than requested duration; no automatic looping")
                for start in range(0, length, 65536):
                    x = source.read(min(65536, length - start), dtype="float32", always_2d=True)
                    out.write(np.repeat(x, 2, axis=1) if source.channels == 1 else x)
        else:
            for start in range(0, length, 65536):
                t = (
                    np.arange(start, min(length, start + 65536), dtype=np.float64)
                    / config.sample_rate
                )
                envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t)
                left = envelope * (
                    0.12 * np.sin(2 * np.pi * 220 * t)
                    + 0.08 * np.sin(2 * np.pi * (440 * t + 11 * t * t))
                )
                right = 0.1 * np.sin(2 * np.pi * 330 * t + 0.3) + left * 0.5
                out.write(np.stack([left, right], axis=1).astype("float32"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--durations", type=float, nargs="+", default=[1, 12, 60])
    parser.add_argument(
        "--input", help="optional real audio; otherwise deterministic stereo synthesis"
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", required=True, help="JSON report (worker: stem directory)")
    parser.add_argument("--worker", choices=["legacy", "bounded"], help=argparse.SUPPRESS)
    parser.add_argument("--max-abs-error", type=float, default=2e-6)
    parser.add_argument("--min-si-sdr", type=float, default=100.0)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    config = ModelConfig(**read_checkpoint(args.checkpoint)["config"])
    report = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1]
        ).strip(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": config.to_dict(),
        "input_kind": "real_audio_prefix" if args.input else "synthetic_tones_chirp_envelope",
        "input_sha256": sha256(args.input) if args.input else None,
        "requested_device": args.device,
        "platform": platform.platform(),
        "threads": args.threads,
        "warmup_windows": args.warmup,
        "scope": "isolated process RSS; file-to-file time; SI-SDR reference is legacy output, not ground truth",
        "thresholds": {"max_abs_error": args.max_abs_error, "min_si_sdr_db": args.min_si_sdr},
        "cases": [],
    }
    success = True
    for seconds in args.durations:
        with tempfile.TemporaryDirectory(prefix="spleeterpt-bounded-bench-") as temp:
            root = Path(temp)
            source = root / "input.wav"
            make_input(source, config, seconds, args.input)
            case = {"seconds": seconds, "input_sha256": sha256(source), "measurements": []}
            for mode in ("legacy", "bounded"):
                command = [
                    sys.executable,
                    __file__,
                    "--worker",
                    mode,
                    "--checkpoint",
                    args.checkpoint,
                    "--device",
                    args.device,
                    "--input",
                    str(source),
                    "--output",
                    str(root / mode),
                    "--threads",
                    str(args.threads),
                    "--warmup",
                    str(args.warmup),
                ]
                completed = subprocess.run(command, check=True, capture_output=True, text=True)
                case["measurements"].append(json.loads(completed.stdout))
            case["stems"] = {
                name: paired_metrics(
                    root / "legacy" / f"{name}.wav", root / "bounded" / f"{name}.wav"
                )
                for name in config.instruments
            }
            case["numerical_pass"] = all(
                m["max_abs_error"] <= args.max_abs_error
                and (
                    m["max_abs_error"] == 0
                    or (
                        m["legacy_reference_si_sdr_db"] is not None
                        and m["legacy_reference_si_sdr_db"] >= args.min_si_sdr
                    )
                )
                for m in case["stems"].values()
            )
            success &= case["numerical_pass"]
            report["cases"].append(case)
            report["numerical_pass"] = success
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f"{seconds:g}s: numerical_pass={case['numerical_pass']}", flush=True)
    if not success:
        raise SystemExit("numerical acceptance threshold failed; see report")


if __name__ == "__main__":
    main()
