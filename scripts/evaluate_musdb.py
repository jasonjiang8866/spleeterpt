"""Reproducible four-way MUSDB evaluation on user-supplied data and checkpoints.

See docs/musdb-evaluation.md. Never downloads data/weights or launches GPU jobs.
"""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from musdb_evaluation import (
    RATE,
    VARIANTS,
    atomic_json,
    decode_track,
    discover,
    score_variants,
    sha256,
    summarize,
    validate_run,
)


def provenance(root):
    root = Path(root).resolve()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    # Include working-tree fingerprints: a HEAD alone does not identify dirty code.
    files = git("ls-files", "-z").split("\0")
    return {
        "root": str(root),
        "commit": git("rev-parse", "HEAD"),
        "status": git("status", "--porcelain"),
        "tracked_sha256": {
            name: sha256(root / name) for name in files if name and (root / name).is_file()
        },
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-output", type=Path, help="validate a completed run without models"
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--layout", choices=("stems", "wav"), default="stems")
    parser.add_argument("--subsets", nargs="+", choices=("train", "test"), default=["test"])
    parser.add_argument(
        "--dataset-label", help="e.g. MUSDB18-HQ; user declaration, not certification"
    )
    parser.add_argument(
        "--track-list", type=Path, help="UTF-8 file: one exact subset/track ID per line"
    )
    parser.add_argument("--limit", type=int, help="default: all selected tracks")
    parser.add_argument("--seed", type=int, help="sample without replacement with --limit")
    parser.add_argument("--expected-tracks", type=int, help="fail unless selection has this count")
    parser.add_argument("--start-seconds", type=float, default=0)
    parser.add_argument("--duration-seconds", type=float, help="default: entire remaining track")
    parser.add_argument("--length-policy", choices=("common-crop", "strict"), default="common-crop")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="device for all three PyTorch variants; TensorFlow stays on CPU",
    )
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--tf-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output", type=Path, help="new run directory; existing directories are refused"
    )
    return parser


def run(args):
    if any(shutil.which(name) is None for name in ("ffmpeg", "ffprobe")):
        raise RuntimeError(
            "museval/stempeg requires ffmpeg and ffprobe on PATH, including WAV evaluation"
        )
    # Set TensorFlow/torch thread controls before their imports. NumPy is already
    # loaded by the helpers, so threadpoolctl also limits its active BLAS pools.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ):
        os.environ[name] = str(args.threads)
    import numpy as np
    import tensorflow as tf
    import torch
    from diagnose_legacy_port import aligned_legacy, load_legacy, use_four_stem_activations
    from threadpoolctl import threadpool_limits
    from verify_audio_parity import OfficialGraph

    from spleeterpt.audio import separate
    from spleeterpt.checkpoint import load_model
    from spleeterpt.config import ModelConfig
    from spleeterpt.convert import import_weights
    from spleeterpt.model import Spleeter

    blas_limit = threadpool_limits(limits=args.threads)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        tf.config.set_visible_devices([], "GPU")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
    tracks = discover(
        args.root,
        args.layout,
        args.subsets,
        args.track_list.read_text().splitlines() if args.track_list else None,
        args.limit,
        args.seed,
    )
    if args.expected_tracks is not None and len(tracks) != args.expected_tracks:
        raise ValueError(f"expected {args.expected_tracks} tracks, selected {len(tracks)}")
    model = load_model(args.checkpoint)
    instruments = list(model.config.instruments)
    if len(instruments) not in (2, 4):
        raise ValueError(
            "MUSDB has no independent piano target; only 2/4-stem evaluation is supported"
        )
    if model.config != ModelConfig.preset(len(instruments)):
        raise ValueError("fair official comparison requires an unmodified 2/4-stem preset")
    checkpoint_files = [Path(str(args.tf_checkpoint) + ext) for ext in (".index", ".meta")]
    checkpoint_files += sorted(args.tf_checkpoint.parent.glob(args.tf_checkpoint.name + ".data-*"))
    if len(checkpoint_files) < 3 or any(not p.is_file() for p in checkpoint_files):
        raise ValueError("TF prefix requires .index, .meta and .data-* files")
    # Verify actual tensors, not just conversion metadata, so a fine-tuned or
    # unrelated checkpoint cannot silently enter an alleged port comparison.
    reference_model = Spleeter(model.config)
    import_weights(tf.train.load_checkpoint(str(args.tf_checkpoint)), reference_model)
    for name, tensor in reference_model.state_dict().items():
        if not torch.equal(tensor, model.state_dict()[name]):
            raise ValueError(f"PyTorch checkpoint differs from supplied TF weights: {name}")
    del reference_model
    model.to(args.device)
    for track in tracks:
        track["sha256"] = {name: sha256(args.root / name) for name in track["files"]}
    manifest = {
        "schema_version": 1,
        "dataset_label": args.dataset_label,
        "scope": "all available selected-subset tracks"
        if not (args.limit or args.track_list)
        else "selected tracks",
        "full_dataset_verified": False,
        "root": str(args.root.resolve()),
        "layout": args.layout,
        "subsets": sorted(set(args.subsets)),
        "selection": tracks,
        "selection_rule": "sort exact IDs; optional NumPy default_rng(seed) sampling; sort again",
        "limit": args.limit,
        "seed": args.seed,
        "expected_tracks": args.expected_tracks,
        "start_seconds": args.start_seconds,
        "duration_seconds": args.duration_seconds,
        "sample_rate": RATE,
        "variants": list(VARIANTS),
        "instruments": instruments,
        "model_config": model.config.to_dict(),
        "length_policy": args.length_policy,
        "metric_settings": {
            "bsseval": "museval.evaluate v4 source images; no permutation; padding=False",
            "filter_samples": 512,
            "window_samples": RATE,
            "hop_samples": RATE,
            "framewise_filters": False,
            "aggregation": "nanmedian frames per track, then nanmedian tracks per target; also mean of track medians",
            "si_sdr": "whole evaluated interval, flatten stereo, remove global mean, one scalar fit",
            "undefined": "null; infinities preserved as +inf/-inf strings; counts reported",
            "bss_tail": "museval 0.4.1 omits incomplete final windows for tracks longer than one second",
        },
        "code": {
            "spleeterpt": provenance(Path(__file__).resolve().parents[1]),
            "upstream": provenance(args.upstream_root),
            "legacy": provenance(args.legacy_root),
        },
        "weight_equivalence": "all converted state tensors exactly match supplied TF checkpoint",
        "weights_sha256": {
            str(p.resolve()): sha256(p) for p in [args.checkpoint, *checkpoint_files]
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "device": args.device,
            "variant_devices": {
                v: "cpu" if v == "tensorflow_current" else args.device for v in VARIANTS
            },
            "cuda": torch.version.cuda if args.device == "cuda" else None,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "tf32": False if args.device == "cuda" else None,
            "packages": {
                p: importlib.metadata.version(p)
                for p in (
                    "torch",
                    "numpy",
                    "scipy",
                    "soundfile",
                    "tensorflow-cpu",
                    "museval",
                    "musdb",
                )
            },
            # Keep the historical report key; capture this interpreter with uv.
            "pip_freeze": subprocess.check_output(
                ["uv", "pip", "freeze", "--python", sys.executable], text=True
            ),
            "threads": args.threads,
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
        },
    }
    if manifest["environment"]["packages"]["museval"] != "0.4.1":
        raise ValueError("this protocol requires museval==0.4.1")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "tracks").mkdir()
    atomic_json(args.output / "manifest.json", manifest)
    status = {
        "state": "running",
        "manifest_sha256": sha256(args.output / "manifest.json"),
        "record_sha256": {},
    }
    atomic_json(args.output / "status.json", status)
    oracle = None
    try:
        legacy = load_legacy(args.legacy_root.resolve(), args.tf_checkpoint, len(instruments))
        corrected = load_legacy(args.legacy_root.resolve(), args.tf_checkpoint, len(instruments))
        if len(instruments) == 4:
            use_four_stem_activations(corrected)
        if args.device == "cuda":
            legacy.to("cuda")
            corrected.to("cuda")
        oracle = OfficialGraph(args.tf_checkpoint, instruments, args.upstream_root)
        records = []
        with torch.inference_mode():
            for index, track in enumerate(tracks):
                # Catch data replacement after the manifest was frozen.
                if any(sha256(args.root / p) != digest for p, digest in track["sha256"].items()):
                    raise ValueError(f"input changed: {track['id']}")
                targets = decode_track(args.root, track, args.layout)
                decoded = targets["mixture"].shape[1]
                start = round(args.start_seconds * RATE)
                stop = (
                    decoded
                    if args.duration_seconds is None
                    else start + round(args.duration_seconds * RATE)
                )
                if start >= decoded or stop > decoded or stop <= start:
                    raise ValueError(f"requested interval unavailable: {track['id']}")
                targets = {name: audio[:, start:stop].copy() for name, audio in targets.items()}
                mixture = torch.from_numpy(targets["mixture"])
                device_mixture = mixture.to(args.device)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                inference_start = time.monotonic()
                variants = {
                    "tensorflow_current": oracle.separate(mixture),
                    "spleeterpt": {n: a.numpy() for n, a in separate(model, mixture).items()},
                    "legacy_unmodified": dict(
                        zip(instruments, [a.cpu().numpy() for a in legacy(device_mixture)])
                    ),
                    "legacy_corrected": dict(
                        zip(
                            instruments,
                            [a.cpu().numpy() for a in aligned_legacy(corrected, device_mixture)],
                        )
                    ),
                }
                record = {
                    "four_variant_inference_seconds": time.monotonic() - inference_start,
                    "id": track["id"],
                    "input_sha256": track["sha256"],
                    "manifest_sha256": status["manifest_sha256"],
                    "decoded_samples": decoded,
                    "segment_start_sample": start,
                    "segment_samples": stop - start,
                    **score_variants(
                        targets, variants, instruments, model.config.hop_length, args.length_policy
                    ),
                }
                filename = f"{index:04d}.json"
                atomic_json(args.output / "tracks" / filename, record)
                status["record_sha256"][filename] = sha256(args.output / "tracks" / filename)
                atomic_json(args.output / "status.json", status)
                records.append(record)
                print(
                    json.dumps({"finished": track["id"], "count": index + 1, "total": len(tracks)}),
                    flush=True,
                )
        atomic_json(args.output / "summary.json", summarize(records, instruments))
        atomic_json(
            args.output / "runtime.json",
            {
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
                if args.device == "cuda"
                else None,
                "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            },
        )
        status["summary_sha256"] = sha256(args.output / "summary.json")
        atomic_json(args.output / "status.json", status)
        validate_run(args.output, require_complete=False)
        status["state"] = "complete"
        atomic_json(args.output / "status.json", status)
        validate_run(args.output)
    except BaseException as exc:
        status.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(args.output / "status.json", status)
        raise
    finally:
        if oracle is not None:
            oracle.close()
        blas_limit.restore_original_limits()


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.validate_output:
        summary = validate_run(args.validate_output)
        print(f"PASS: complete results for {summary['track_count']} tracks (not a quality verdict)")
        return
    for name in (
        "root",
        "dataset_label",
        "upstream_root",
        "legacy_root",
        "tf_checkpoint",
        "checkpoint",
        "output",
    ):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required for evaluation")
    import math

    if args.threads < 1 or not math.isfinite(args.start_seconds) or args.start_seconds < 0:
        parser.error("positive threads and nonnegative finite start required")
    if args.duration_seconds is not None and (
        not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0
    ):
        parser.error("duration must be positive and finite")
    run(args)


if __name__ == "__main__":
    main()
