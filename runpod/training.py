"""Real train-pool preview acceptance; not a convergence or generalization study."""

import argparse
import gc
import json
import os
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runpod"))
from quality import command, digest, download  # noqa: E402 -- script-local module path


def equal_state(a, b):
    import torch

    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict) and a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)):
        return (
            type(a) is type(b) and len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
        )
    return type(a) is type(b) and a == b


def phase(argv, output):
    import torch

    from spleeterpt.cli import main

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    main(argv)
    torch.cuda.synchronize()
    output.write_text(
        json.dumps(
            {
                "seconds": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        )
    )


def prepare_preview(folder):
    import numpy as np
    import soundfile as sf

    folder.mkdir(exist_ok=False)
    spec = json.loads((ROOT / "runpod/training-assets.json").read_text())
    archive = download("preview.zip", folder)
    decoded = {}
    with zipfile.ZipFile(archive) as z:
        for split, tracks in spec.items():
            paths = []
            for name, expected in tracks.items():
                source = folder / Path(name).name
                source.write_bytes(z.read(name))
                if digest(source) != expected:
                    raise ValueError("Training asset checksum mismatch")
                track = folder / split / source.name.removesuffix(".stem.mp4")
                track.mkdir(parents=True)
                shapes = []
                hashes = {}
                for index, stem in enumerate(("mixture", "drums", "bass", "other", "vocals")):
                    dest = track / f"{stem}.wav"
                    command(
                        "ffmpeg",
                        "-v",
                        "error",
                        "-i",
                        source,
                        "-map",
                        f"0:a:{index}",
                        "-c:a",
                        "pcm_f32le",
                        dest,
                    )
                    audio, rate = sf.read(dest, dtype="float32", always_2d=True)
                    if rate != 44100 or audio.shape[1] != 2 or not np.isfinite(audio).all():
                        raise ValueError("Unexpected decoded audio")
                    shapes.append(audio.shape)
                    hashes[stem] = digest(dest)
                if len(set(shapes)) != 1 or not 6 < shapes[0][0] / 44100 < 8:
                    raise ValueError("Unaligned or unexpected preview duration")
                decoded[name] = {"samples": shapes[0][0], "wav_sha256": hashes}
                paths.append(str(track))
            (folder / f"{split}.jsonl").write_text(
                "".join(json.dumps({"track": p}) + "\n" for p in paths)
            )
    return spec, decoded


def run():
    import torch

    from spleeterpt.audio import read_audio, separate
    from spleeterpt.checkpoint import load_model, read_checkpoint
    from spleeterpt.config import ModelConfig

    if torch.__version__ != "2.8.0+cu128" or not torch.cuda.is_available():
        raise RuntimeError("Stable torch 2.8.0+cu128 and CUDA required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    folder = Path("/workspace/training-preview")
    spec, decoded = prepare_preview(folder)
    reports = []
    for stems in (2, 4):
        weights = folder / f"{stems}stems"
        weights.mkdir()
        archive = download(f"{stems}stems.tar.gz", folder)
        with tarfile.open(archive) as tar:
            tar.extractall(weights, filter="data")
        initial_path = folder / f"{stems}stems.pt"
        command(
            sys.executable,
            "-m",
            "spleeterpt.cli",
            "convert",
            "--checkpoint",
            weights / "model",
            "--stems",
            stems,
            "--output",
            initial_path,
        )
        common = [
            "train",
            "--config",
            str(ROOT / f"configs/{stems}stems.json"),
            "--train-manifest",
            str(folder / "train.jsonl"),
            "--val-manifest",
            str(folder / "validation.jsonl"),
            "--device",
            "cuda",
            "--precision",
            "fp32",
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
            "--epochs",
            "2",
            "--lr",
            "1e-5",
            "--seed",
            "17",
            "--deterministic",
        ]
        uninterrupted = folder / f"{stems}-continuous"
        resumed = folder / f"{stems}-resumed"
        stages = [
            ("continuous", uninterrupted, ["--init", str(initial_path)]),
            ("interrupted", resumed, ["--init", str(initial_path), "--max-steps", "3"]),
            ("resumed", resumed, ["--resume", str(resumed / "last.pt")]),
        ]
        timings = {}
        interrupted_cursor = None
        for label, out, extra in stages:
            timing = folder / f"{stems}-{label}.json"
            command(
                sys.executable,
                __file__,
                "--phase-output",
                timing,
                "--",
                *common,
                "--output",
                out,
                *extra,
            )
            timings[label] = json.loads(timing.read_text())
            if label == "interrupted":
                cp = read_checkpoint(out / "last.pt")
                interrupted_cursor = [cp["step"], cp["epoch"], cp["next_batch"]]
                del cp
        a, b = (read_checkpoint(p / "last.pt") for p in (uninterrupted, resumed))
        initial = read_checkpoint(initial_path)
        expected_config = ModelConfig.preset(stems).to_dict()
        state_keys = (
            "model",
            "optimizer",
            "scaler",
            "rng_cpu",
            "rng_cuda",
            "signature",
            "step",
            "epoch",
            "next_batch",
            "best",
            "config",
        )
        equality = {k: equal_state(a[k], b[k]) for k in state_keys}
        metrics = {
            label: [json.loads(line) for line in (p / "metrics.jsonl").read_text().splitlines()]
            for label, p in (("continuous", uninterrupted), ("resumed", resumed))
        }
        best_a, best_b = (read_checkpoint(p / "best.pt") for p in (uninterrupted, resumed))
        best_equal = all(equal_state(best_a[k], best_b[k]) for k in state_keys)
        del best_a, best_b
        model = load_model(resumed / "best.pt", "cuda")
        validation_track = Path(json.loads((folder / "validation.jsonl").read_text())["track"])
        audio = read_audio(validation_track / "mixture.wav", 44100)
        estimates = separate(model, audio)
        inference_ok = set(estimates) == set(model.config.instruments) and all(
            x.shape == audio.shape and torch.isfinite(x).all().item() for x in estimates.values()
        )
        record = {
            "stems": stems,
            "config": a["config"],
            "expected_config_matches": a["config"] == expected_config,
            "precision": "fp32",
            "lr": 1e-5,
            "batch_size": 1,
            "epochs": 2,
            "interrupted_cursor": interrupted_cursor,
            "final_cursor": [b["step"], b["epoch"], b["next_batch"]],
            "state_equal": equality,
            "best_state_equal": best_equal,
            "metrics": metrics,
            "metrics_equal": metrics["continuous"] == metrics["resumed"],
            "weights_changed": any(
                not torch.equal(initial["model"][k], v) for k, v in b["model"].items()
            ),
            "weights_finite": all(torch.isfinite(v).all().item() for v in b["model"].values()),
            "optimizer_steps": sorted(
                {int(v["step"].item()) for v in b["optimizer"]["state"].values()}
            ),
            "inference_finite": inference_ok,
            "timings": timings,
            "initial_sha256": digest(initial_path),
            "last_sha256": digest(resumed / "last.pt"),
            "best_sha256": digest(resumed / "best.pt"),
        }
        reports.append(record)
        print(f"Completed {stems}-stem training and fresh-process resume comparison", flush=True)
        del a, b, initial, model, estimates
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "kind": "training-preview",
        "commit": os.environ["SPT_COMMIT"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "assets": spec,
        "decoded": decoded,
        "runs": reports,
        "seconds": time.monotonic() - started,
        "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase-output", type=Path)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.phase_output:
        phase(args.training_args[1:], args.phase_output)
    else:
        from training_result import validate_training

        report = run()
        result = {"status": "passed", "report": report}
        # Keep the detailed evidence even when acceptance fails.
        args.output.write_text(json.dumps(report, indent=2))
        validate_training(result, os.environ["SPT_COMMIT"])
