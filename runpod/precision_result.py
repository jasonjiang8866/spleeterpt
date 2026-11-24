"""Audit the CUDA precision screen after recovery, using only the standard library."""

import gzip
import itertools
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs/results/cuda-settings-20261123/cuda-settings.json.gz"
CONTROL = "no_tf32+deterministic"
MODES = ("full", "bounded")
STEMS = {"vocals", "piano", "bass", "drums", "other"}
OPS = ("fft64", "decoder64", "output64")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def waveform_pass(row, frames):
    error, rms, ref = (row[k] for k in ("max_abs_error", "rms_error", "reference_rms"))
    score = row["legacy_reference_si_sdr_db"]
    if row["samples_per_channel"] != frames or not all(
        finite(v) and v >= 0 for v in (error, rms, ref)
    ):
        raise ValueError("Invalid waveform evidence")
    if score is not None and not finite(score):
        raise ValueError("Invalid SI-SDR")
    if ref <= 1e-5:
        return rms <= 1e-8 and error <= 1e-7
    return error <= 2e-6 and (error == 0 or (score is not None and score >= 100))


def audit(study):
    with gzip.open(BASELINE) as stream:
        baseline = json.load(stream)
    profiles = {
        "+".join((CONTROL, *ops)): list(ops)
        for size in range(4)
        for ops in itertools.combinations(OPS, size)
    }
    order = [(1, name) for name in profiles] + [(2, name) for name in reversed(profiles)]
    cases = study["cases"]
    if (
        study["checkpoint_sha256"] != baseline["checkpoint_sha256"]
        or set(cases) != {"music_original", "real60"}
        or set(study["summary"]) != set(profiles)
    ):
        raise ValueError("Precision checkpoint or coverage mismatch")

    candidates = set(profiles)
    control_pcm = {}
    control_counts = {}
    for name, case in cases.items():
        original = name == "music_original"
        frames = 479832 if original else 60 * 44100
        runs = case["runs"]
        previous = baseline["cases"][name]
        if (
            case["input_pcm_sha256"] != previous["input_pcm_sha256"]
            or [(r["repeat"], r["profile"]) for r in runs] != order
        ):
            raise ValueError("Precision input or execution order mismatch")
        if original and (
            set(case["reference_sha256"]) != STEMS
            or not all(re.fullmatch("[0-9a-f]{64}", h) for h in case["reference_sha256"].values())
        ):
            raise ValueError("Invalid TF reference hashes")

        for run in runs:
            profile = run["profile"]
            settings = run["settings"]
            if (
                settings["fp64_operations"] != profiles[profile]
                or settings["cudnn_deterministic"] is not True
                or settings["cudnn_tf32"] is not False
                or settings["matmul_tf32"] is not False
                or settings["warmup_windows"] != 0
            ):
                raise ValueError("Precision configuration mismatch")
            hashes = run["output_pcm_sha256"]
            if (
                set(hashes) != set(MODES)
                or any(
                    set(stems) != STEMS
                    or not all(re.fullmatch("[0-9a-f]{64}", h) for h in stems.values())
                    for stems in hashes.values()
                )
                or set(run["bounded_vs_full"]) != STEMS
            ):
                raise ValueError("Missing precision outputs")
            if original != ("vs_tf" in run):
                raise ValueError("Incorrect TF comparison coverage")
            for row in run["bounded_vs_full"].values():
                passed = waveform_pass(row, frames)
                if passed == row["verdict"].startswith("failed"):
                    raise ValueError("Stored waveform verdict disagrees with measurements")
                if not passed:
                    candidates.discard(profile)
            if original:
                tf = run["vs_tf"]
                if set(tf) != set(MODES) or any(set(stems) != STEMS for stems in tf.values()):
                    raise ValueError("Missing TF comparisons")
                for mode, metrics in tf.items():
                    for row in metrics.values():
                        count, ratio = row["strict_failed_samples"], row["max_tolerance_ratio"]
                        if (
                            type(count) is not int
                            or not 0 <= count <= 2 * frames
                            or not finite(ratio)
                            or ratio < 0
                            or (count == 0) != (ratio <= 1)
                        ):
                            raise ValueError("Inconsistent TF failures and tolerance ratio")
                        if count:
                            candidates.discard(profile)
                    if profile == CONTROL:
                        control_counts.setdefault(mode, []).append(
                            {stem: row["strict_failed_samples"] for stem, row in metrics.items()}
                        )

        for first, second in zip(runs[: len(profiles)], reversed(runs[len(profiles) :])):
            same_pcm = first["output_pcm_sha256"] == second["output_pcm_sha256"]
            if not same_pcm:
                candidates.discard(first["profile"])
            if first["profile"] == CONTROL:
                historical = next(r for r in previous["runs"] if r["profile"] == CONTROL)
                control_pcm[name] = (
                    same_pcm and first["output_pcm_sha256"] == historical["output_pcm_sha256"]
                )

    screened = [p for p in profiles if p in candidates]
    if screened != study["screen_passed_profiles"]:
        raise ValueError("Stored precision candidates disagree with raw evidence")
    return {
        "screen_passed_profiles": screened,
        "control_tf_failed_samples": control_counts,
        "control_pcm_matches_previous": control_pcm,
        "tf_reference_matches_previous": cases["music_original"]["reference_sha256"]
        == baseline["cases"]["music_original"]["reference_sha256"],
    }
