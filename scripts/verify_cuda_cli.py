"""Compare the selected CUDA CLI profile with its full-file path and the frozen TF CPU graph."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
from benchmark_bounded import acceptance, make_input, paired_metrics

from spleeterpt.audio import read_audio
from spleeterpt.config import ModelConfig
from spleeterpt.pretrained import digest, verified_download

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ModelConfig.preset(5)
MODES = ("full", "bounded")


def pcm_hash(data):
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def cli_args(checkpoint, source, root, mode, *, profile="default"):
    args = ["separate", str(source), "--checkpoint", str(checkpoint), "--device", "cuda"]
    args += ["--output", str(root / mode)]
    if profile == "fft64":
        args.append("--five-stem-cuda-precision")
    elif profile != "default":
        raise ValueError(f"Unknown CUDA CLI profile: {profile}")
    if mode == "bounded":
        args.append("--bounded-memory")
    return args


def cli(checkpoint, source, root, mode, *, profile):
    # Call the installed CLI entry point in a fresh process so each peak includes model loading.
    runner = """import json, sys, torch
from spleeterpt.cli import main
main(sys.argv[1:])
torch.cuda.synchronize()
print(json.dumps(torch.cuda.max_memory_allocated() / 2**20))
"""
    peak = subprocess.check_output(
        [sys.executable, "-c", runner, *cli_args(checkpoint, source, root, mode, profile=profile)],
        text=True,
    )
    return json.loads(peak.splitlines()[-1])


def bounded_metrics(root, frames, thresholds):
    rows = {}
    for stem in CONFIG.instruments:
        file = stem + ".wav"
        metrics = paired_metrics(root / "full" / file, root / "bounded" / file)
        if metrics["samples_per_channel"] != frames:
            raise ValueError("Incorrect output duration")
        metrics["verdict"] = acceptance(metrics, thresholds)
        if metrics["reference_rms"] <= 1e-5 and (
            metrics["rms_error"] > 1e-8 or metrics["max_abs_error"] > 1e-7
        ):
            metrics["verdict"] = "failed_near_silent"
        rows[stem] = metrics
    return rows


def replay(work, destination):
    from parity_metrics import agreement_with_failures, multiwindow_failure_locations
    from verify_audio_parity import OfficialGraph, audio_cases
    from verify_native_five_stem import passed, score_holdout

    pins = json.loads((ROOT / "scripts/five_stem_source.json").read_text())
    history = json.loads(
        (ROOT / "docs/results/five-stem-20261102/5stems-strict-final.json").read_text()
    )
    weights = work / "weights"
    checkpoint = weights / "5stems.pt"
    tf_prefix = weights / "5stems/model"
    upstream = work / "upstream"
    subprocess.run(
        ["git", "clone", "-q", "https://github.com/deezer/spleeter.git", upstream], check=True
    )
    subprocess.run(
        ["git", "-C", upstream, "checkout", "--detach", pins["upstream_commit"]], check=True
    )
    source = upstream / "audio_example.mp3"
    real = work / "real.wav"
    old = json.loads(
        (ROOT / "docs/results/five-stem-gpu-20261102/41477211252/result.json").read_text()
    )
    if (digest(source), digest(checkpoint), digest(real)) != (
        pins["audio_sha256"],
        pins["converted_sha256"],
        old["report"]["real_input"]["sha256"],
    ) or any(
        digest(Path(str(tf_prefix) + suffix)) != value
        for suffix, value in history["tf_checkpoint_files"].items()
    ):
        raise ValueError("Frozen input or model mismatch")
    profile = os.environ.get("SPT_CUDA_CLI_PROFILE", "default")
    report = {
        "cli_profile": profile,
        "calibration": {},
        "long_audio": {},
        "trios": None,
        "checkpoint_sha256": digest(checkpoint),
        "upstream_commit": pins["upstream_commit"],
        "reference": "official TF CPU, oneDNN disabled",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }

    def save():
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    thresholds = SimpleNamespace(max_abs_error=2e-6, min_si_sdr=100.0)
    for case, seconds, original in (
        ("real60", 60, real),
        ("synthetic60", 60, None),
        ("synthetic600", 600, None),
    ):
        with tempfile.TemporaryDirectory(dir=work) as tmp:
            root = Path(tmp)
            wav = root / "input.wav"
            make_input(wav, CONFIG, seconds, original)
            peaks = {mode: cli(checkpoint, wav, root, mode, profile=profile) for mode in MODES}
            rows = bounded_metrics(root, seconds * 44100, thresholds)
            report["long_audio"][case] = {"peak_allocated_mib": peaks, "stems": rows}
            save()

    def read_outputs(root, mode, frames):
        out = {}
        for stem in CONFIG.instruments:
            data, rate = sf.read(root / mode / (stem + ".wav"), dtype="float32", always_2d=True)
            if (data.shape, rate) != ((frames, 2), 44100) or not np.isfinite(data).all():
                raise ValueError("Invalid CLI output")
            out[stem] = torch.from_numpy(np.ascontiguousarray(data.T))
        return out

    precision = os.environ.get("SPT_CUDA_PRECISION_STUDY") == "true"
    settings_requested = precision or os.environ.get("SPT_CUDA_SETTINGS_STUDY") == "true"
    calibration_pair = None
    oracle = OfficialGraph(tf_prefix, CONFIG.instruments, upstream)
    try:
        original_music = read_audio(source, 44100)
        for name, audio in audio_cases(original_music, CONFIG, True).items():
            with tempfile.TemporaryDirectory(dir=work) as tmp:
                root = Path(tmp)
                wav = root / "input.wav"
                sf.write(wav, audio.numpy().T, 44100, subtype="FLOAT")
                key = pcm_hash(read_audio(wav, 44100).numpy())
                if key != history["case_provenance"][name]["decoded_pcm_sha256"]:
                    raise ValueError("Calibration PCM changed")
                ref = oracle.separate(audio)
                if settings_requested and name == "music_original":
                    calibration_pair = audio, ref
                case = {
                    "input_pcm_sha256": key,
                    "output_pcm_sha256": {},
                    "reference_sha256": {s: pcm_hash(a) for s, a in ref.items()},
                }
                failure_masks = {}
                for mode in MODES:
                    cli(checkpoint, wav, root, mode, profile=profile)
                    outputs = read_outputs(root, mode, audio.shape[1])
                    case["output_pcm_sha256"][mode] = {
                        s: pcm_hash(a.numpy()) for s, a in outputs.items()
                    }
                    case[mode] = {}
                    for stem, output in outputs.items():
                        row, failures = agreement_with_failures(ref[stem], output.numpy())
                        case[mode][stem] = row
                        if name == "music_multiwindow":
                            failure_masks.setdefault(mode, {})[stem] = failures
                if failure_masks:
                    case["failure_locations"] = multiwindow_failure_locations(
                        failure_masks["full"],
                        failure_masks["bounded"],
                        CONFIG,
                        audio.shape[-1],
                        original_music.shape[-1],
                    )
                report["calibration"][name] = case
                save()

        if settings_requested:
            if calibration_pair is None:
                raise ValueError("Missing original music calibration")
            from study_cuda_settings import study

            study(
                checkpoint,
                *calibration_pair,
                real,
                thresholds,
                destination.with_name("cuda-precision.json" if precision else "cuda-settings.json"),
                report["source_commit"],
                precision=precision,
            )

        def bounded(audio):
            with tempfile.TemporaryDirectory(dir=work) as tmp:
                root = Path(tmp)
                wav = root / "mix.wav"
                sf.write(wav, audio.numpy().T, 44100, subtype="FLOAT")
                cli(checkpoint, wav, root, "bounded", profile=profile)
                return read_outputs(root, "bounded", audio.shape[1])

        long_audio = report["long_audio"]
        short = long_audio["synthetic60"]["peak_allocated_mib"]["bounded"]
        long = long_audio["synthetic600"]["peak_allocated_mib"]
        stream_pass = all(
            not row["verdict"].startswith("failed")
            for case in long_audio.values()
            for row in case["stems"].values()
        )
        if (
            all(passed(c[m]) for c in report["calibration"].values() for m in MODES)
            and stream_pass
            and long["bounded"] <= short * 1.1 + 32
            and long["bounded"] < long["full"] * 0.5
        ):
            manifest = json.loads((ROOT / "scripts/five_stem_corpus.json").read_text())
            spec = manifest["trios_archive"]
            archive = verified_download(
                spec["url"], work / "trios.zip", sha256=spec["sha256"], size=spec["bytes"]
            )
            report["trios"] = score_holdout(archive, manifest, oracle, bounded)
    finally:
        oracle.close()
        save()


if __name__ == "__main__":
    if os.environ.get("TF_ENABLE_ONEDNN_OPTS") != "0":
        raise ValueError("Disable oneDNN before starting the reference process")
    replay(Path(sys.argv[1]), Path(sys.argv[2]))
