"""Paired CUDA backend ablation on the original music and historical 60-second mix."""

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import patch

import soundfile as sf
import torch
from benchmark_bounded import make_input
from parity_metrics import agreement
from verify_cuda_cli import CONFIG, MODES, bounded_metrics, cli_args, pcm_hash

from spleeterpt import audio, cli, streaming
from spleeterpt.audio import read_audio, separate

# Each contrast changes one factor. Also run the complete benchmark configuration.
FACTORS = ("no_tf32", "deterministic", "warmup")
PROFILES = [
    "+".join(subset) or "default"
    for size in range(len(FACTORS) + 1)
    for subset in itertools.combinations(FACTORS, size)
] + ["benchmark"]


# Each precision arm inherits the previously measured deterministic FP32 control.
WIDE_OPS = ("fft64", "decoder64", "output64")
PRECISION_PROFILES = [
    "+".join(("no_tf32", "deterministic", *subset))
    for size in range(len(WIDE_OPS) + 1)
    for subset in itertools.combinations(WIDE_OPS, size)
]


def run_cli(profile, args):
    factors = set(FACTORS if profile == "benchmark" else profile.split("+"))
    if profile == "benchmark":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(2)
        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
    if "no_tf32" in factors:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if "deterministic" in factors:
        torch.backends.cudnn.deterministic = True

    load = cli.load_model

    def study_load(checkpoint, device):
        model = load(checkpoint, device)
        if len(model.config.instruments) != 5:
            raise ValueError("CUDA study requires a five-stem checkpoint")
        for net in model.unets.values():
            if "decoder64" in factors:
                net.decoder[-1].forward = partial(net.decoder[-1].forward, wide=True)
            if "output64" in factors:
                net.output.forward = partial(net.output.forward, wide=True)
        if "warmup" in factors:
            separate(model, torch.zeros(2, model.config.hop_length))
            torch.cuda.synchronize()
        return model

    with ExitStack() as stack:
        if factors.intersection(("warmup", "decoder64", "output64")):
            stack.enter_context(patch.object(cli, "load_model", study_load))
        if "fft64" in factors:
            wide_stft = partial(audio.stft, wide_fft=True)
            # Both entry points import the same STFT under separate module names.
            stack.enter_context(patch.object(audio, "stft", wide_stft))
            stack.enter_context(patch.object(streaming, "stft", wide_stft))
        cli.main(args)
    torch.cuda.synchronize()

    settings = {
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "threads": torch.get_num_threads(),
        "fp64_operations": [op for op in WIDE_OPS if op in factors],
        "warmup_windows": int("warmup" in factors),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(),
    }
    print(json.dumps(settings))


def measure_pair(profile, checkpoint, wav, root, reference, thresholds):
    frames = sf.info(wav).frames
    settings = {}
    hashes = {}
    tf = {}
    for mode in MODES:
        command = [sys.executable, __file__, profile, *cli_args(checkpoint, wav, root, mode)]
        settings[mode] = json.loads(subprocess.check_output(command, text=True).splitlines()[-1])
        hashes[mode] = {}
        tf[mode] = {}
        for stem in CONFIG.instruments:
            data, rate = sf.read(root / mode / (stem + ".wav"), dtype="float32", always_2d=True)
            if data.shape != (frames, 2) or rate != CONFIG.sample_rate:
                raise ValueError(f"Invalid {mode} output for {stem}")
            hashes[mode][stem] = pcm_hash(data.T)
            if reference is not None:
                if reference[stem].shape != data.T.shape:
                    raise ValueError(f"Reference shape mismatch for {stem}")
                metric = agreement(reference[stem], data.T)
                tf[mode][stem] = {
                    k: metric[k]
                    for k in ("strict_failed_samples", "max_absolute_error", "max_tolerance_ratio")
                }
    if settings["full"] != settings["bounded"]:
        raise ValueError("Unpaired CUDA settings")
    result = {
        "settings": settings["full"],
        "output_pcm_sha256": hashes,
        "bounded_vs_full": bounded_metrics(root, frames, thresholds),
    }
    if reference is not None:
        result["vs_tf"] = tf
    return result


def summarize(cases, profiles):
    summary = {}
    for profile in profiles:
        row = {}
        for name, case in cases.items():
            runs = [run for run in case["runs"] if run["profile"] == profile]
            if [run["repeat"] for run in runs] != [1, 2]:
                raise ValueError(f"Incomplete {name}/{profile} repeats")
            paired = [r["bounded_vs_full"] for r in runs]
            row[name] = {
                "identical_repeat_pcm": runs[0]["output_pcm_sha256"]
                == runs[1]["output_pcm_sha256"],
                "bounded_stems_passed": [
                    sum(not m["verdict"].startswith("failed") for m in stems.values())
                    for stems in paired
                ],
                "worst_bounded_max_abs_error": max(
                    m["max_abs_error"] for stems in paired for m in stems.values()
                ),
            }
            if name == "music_original":
                row[name]["tf_stems_passed"] = {
                    mode: [
                        sum(m["strict_failed_samples"] == 0 for m in r["vs_tf"][mode].values())
                        for r in runs
                    ]
                    for mode in MODES
                }
        summary[profile] = row
    return summary


def study(
    checkpoint,
    original,
    original_ref,
    real,
    thresholds,
    destination,
    source_commit,
    *,
    precision=False,
):
    from spleeterpt.pretrained import digest

    profiles = PRECISION_PROFILES if precision else PROFILES
    cases = {}
    report = {
        "source_commit": source_commit,
        "study_kind": "precision" if precision else "backend",
        "checkpoint_sha256": digest(checkpoint),
        "release_eligible": False,
        "complete": False,
        "cases": cases,
    }

    def save():
        pending = destination.with_suffix(".tmp")
        pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

        pending.replace(destination)

    save()
    with tempfile.TemporaryDirectory(dir=checkpoint.parent) as tmp:
        directory = Path(tmp)
        for name in ("music_original", "real60"):
            path = directory / (name + ".wav")
            if name == "music_original":
                sf.write(path, original.numpy().T, CONFIG.sample_rate, subtype="FLOAT")
                if not torch.equal(read_audio(path, CONFIG.sample_rate), original):
                    raise ValueError("Study input differs from frozen calibration")
                reference = original_ref
            else:
                make_input(path, CONFIG, 60, real)
                reference = None
            runs = []
            cases[name] = {
                "input_pcm_sha256": pcm_hash(read_audio(path, CONFIG.sample_rate).numpy()),
                "reference_sha256": {s: pcm_hash(a) for s, a in reference.items()}
                if reference
                else None,
                "runs": runs,
            }
            for repeat, order in enumerate((profiles, reversed(profiles)), start=1):
                for profile in order:
                    with tempfile.TemporaryDirectory(dir=directory) as outputs:
                        pair = measure_pair(
                            profile, checkpoint, path, Path(outputs), reference, thresholds
                        )
                    runs.append({"repeat": repeat, "profile": profile, **pair})
                    save()
    report["summary"] = summarize(cases, profiles)
    if precision:
        # A screen can nominate a configuration only for the next, full validation.
        expected = [len(CONFIG.instruments)] * 2
        report["screen_passed_profiles"] = [
            profile
            for profile, row in report["summary"].items()
            if all(case["identical_repeat_pcm"] for case in row.values())
            and all(case["bounded_stems_passed"] == expected for case in row.values())
            and all(row["music_original"]["tf_stems_passed"][mode] == expected for mode in MODES)
        ]
    report["complete"] = True
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=sorted(set(PROFILES + PRECISION_PROFILES)))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    run_cli(args.profile, args.args)
