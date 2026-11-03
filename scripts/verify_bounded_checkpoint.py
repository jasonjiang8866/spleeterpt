"""Official-checkpoint file-path edge checks, without TensorFlow or paid resources.

Use benchmark_bounded.py separately for isolated-process memory measurements.
SI-SDR reference is the legacy output; no ground-truth quality claim is made.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
from benchmark_bounded import acceptance, paired_metrics
from fft_batch_policy import fft_batch_policy

from spleeterpt.audio import read_audio, separate
from spleeterpt.checkpoint import load_model
from spleeterpt.pretrained import digest
from spleeterpt.streaming import separate_file


def cases(config):
    """Both padded model-window transition and nominal window/hop/FFT edges."""
    lengths = {1}
    for edge in (
        config.hop_length,
        config.n_fft,
        config.hop_length * config.segment_frames - config.n_fft,
        config.hop_length * config.segment_frames,
    ):
        lengths.update((edge - 1, edge, edge + 1))
    rng = np.random.default_rng(1729)
    for length in sorted(lengths):
        audio = rng.normal(0, 0.03, (length, 2)).astype(np.float32)
        audio[[0, length // 2, -1]] = [0.2, -0.15]
        yield f"boundary_{length}", audio
    yield "silence", np.zeros((config.n_fft + 1, 2), dtype=np.float32)
    yield "mono", rng.normal(0, 0.03, (config.n_fft + 1, 1)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--fft-batch-policy", choices=["native", "pad-long-tail"], default="native")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = load_model(args.checkpoint, args.device)
    report = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_modified": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "checkpoint_sha256": digest(args.checkpoint),
        "config": model.config.to_dict(),
        "fft_batch_policy": args.fft_batch_policy,
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        },
        "scope": "legacy-reference agreement; deterministic synthetic edges, no music quality evidence; timing excludes loading, includes file I/O; no isolated memory measurement",
        "thresholds": {
            "max_abs_error": 2e-6,
            "min_si_sdr_db": 100,
            "near_silent_reference_rms": 1e-5,
            "near_silent_rms_error": 1e-8,
            "near_silent_max_abs_error": 1e-7,
        },
        "cases": {},
        "passed": True,
    }
    limits = SimpleNamespace(max_abs_error=2e-6, min_si_sdr=100)
    for name, audio in cases(model.config):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.wav"
            sf.write(source, audio, model.config.sample_rate, subtype="FLOAT")
            started = time.perf_counter()
            legacy = separate(model, read_audio(source, model.config.sample_rate))
            for stem, value in legacy.items():
                sf.write(
                    root / f"{stem}.wav", value.numpy().T, model.config.sample_rate, subtype="FLOAT"
                )
            legacy_seconds = time.perf_counter() - started
            del legacy
            started = time.perf_counter()
            with fft_batch_policy(args.fft_batch_policy, model.config, len(audio)):
                separate_file(model, source, root / "bounded")
            bounded_seconds = time.perf_counter() - started
            metrics = {
                stem: paired_metrics(root / f"{stem}.wav", root / "bounded" / f"{stem}.wav")
                for stem in model.config.instruments
            }
            for value in metrics.values():
                value["acceptance"] = acceptance(value, limits)
            passed = all(not v["acceptance"].startswith("failed") for v in metrics.values())
            report["cases"][name] = {
                "samples": len(audio),
                "channels": audio.shape[1],
                "input_sha256": digest(source),
                "pcm_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                "legacy_seconds": legacy_seconds,
                "bounded_seconds": bounded_seconds,
                "stems": metrics,
                "passed": passed,
            }
            report["passed"] &= passed
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(f"{name}: passed={passed}", flush=True)
    if not report["passed"]:
        raise SystemExit("FAIL: official-checkpoint bounded edge comparison")


if __name__ == "__main__":
    main()
