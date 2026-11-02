"""Independent acceptance of actual GPU measurements; no Torch dependency."""

import json
import math
import re

from gpu_profile import MIG24, validate_runtime
from quality import ROOT


def number(x, *, positive=False):
    return (
        not isinstance(x, bool)
        and isinstance(x, (int, float))
        and math.isfinite(x)
        and (x > 0 if positive else x >= 0)
    )


def validate(result, sha, volume, pod, experiment, *, five_stem=False):
    report = result.get("report", {})
    if (
        result.get("status") != "passed"
        or report.get("complete") is not True
        or report.get("kind") != ("five-stem-bounded-v1" if five_stem else "bounded-memory-v1")
        or report.get("commit") != sha
        or report.get("global_volume_id") != volume
        or report.get("pod_id") != pod
        or report.get("experiment") != experiment
    ):
        raise ValueError("Incomplete or mismatched bounded benchmark identity")
    validate_runtime(report["gpu_runtime"], MIG24)
    if (
        report["cuda_tests"]["returncode"] != 0
        or "3 passed" not in report["cuda_tests"]["output"]
        or report["real_input"]["test_pool_used"] is not False
    ):
        raise ValueError("CUDA tests or real input provenance incomplete")
    released = json.loads((ROOT / "src/spleeterpt/pretrained.json").read_text())["models"]
    if five_stem:
        pinned = json.loads((ROOT / "scripts/five_stem_source.json").read_text())
        released = {"5stems": {"sha256": pinned["converted_sha256"]}}
        if (
            report["conversion"]["source"] != pinned
            or report["conversion"]["checkpoint_sha256"] != pinned["converted_sha256"]
        ):
            raise ValueError("Official five-stem provenance mismatch")
        validate_edges(report["edges"], sha, pinned["converted_sha256"])
    expected_keys = {
        f"{k}stems-{kind}"
        for k in ((5,) if five_stem else (2, 4))
        for kind in ("synthetic", "real")
    }
    if five_stem:
        piano = json.loads((ROOT / "scripts/piano_source.json").read_text())
        if report["piano_source"] != piano:
            raise ValueError("Unpinned piano source")
        expected_keys.update(f"5stems-piano-{start}" for start in (0, 15, 30))
    if set(report["benchmarks"]) != expected_keys:
        raise ValueError("Missing checkpoint/input combinations")
    for key, bench in report["benchmarks"].items():
        model, kind = key.split("-", 1)
        stems = (
            ("vocals", "accompaniment")
            if model == "2stems"
            else ("vocals", "piano", "drums", "bass", "other")
            if model == "5stems"
            else ("vocals", "drums", "bass", "other")
        )
        if (
            bench["checkpoint_sha256"] != released[model]["sha256"]
            or bench["source_commit"] != sha
            or bench["requested_device"] != "cuda"
            or bench["numerical_pass"] is not True
            or bench["config"]["instruments"] != list(stems)
            or bench["thresholds"]
            != {
                "max_abs_error": 2e-6,
                "min_si_sdr_db": 100.0,
                "near_silent_reference_rms": 1e-5,
                "near_silent_rms_error": 1e-8,
                "near_silent_max_abs_error": 1e-7,
            }
        ):
            raise ValueError("Checkpoint/protocol mismatch")
        for name, value in {
            "n_fft": 4096,
            "hop_length": 1024,
            "sample_rate": 44100,
            "segment_frames": 512,
            "frequency_bins": 1024,
            "mask_extension": "zeros",
            "separation_exponent": 2.0,
        }.items():
            if bench["config"][name] != value:
                raise ValueError("Acoustic configuration changed")
        if kind == "real" and bench["input_sha256"] != report["real_input"]["sha256"]:
            raise ValueError("Real audio checksum mismatch")
        if kind.startswith("piano-"):
            item = report["piano_inputs"][key]
            if (
                item["start_seconds"] != int(kind.split("-")[1])
                or item["samples"] != 12 * 44100
                or bench["input_sha256"] != item["sha256"]
            ):
                raise ValueError("Piano clip input mismatch")
        durations = (
            [12]
            if kind.startswith("piano-")
            else [1, 12, 60, 600]
            if kind == "synthetic"
            else [1, 12, 60]
        )
        if [c["seconds"] for c in bench["cases"]] != durations:
            raise ValueError("Missing requested durations")
        for case in bench["cases"]:
            if not re.fullmatch(r"[a-f0-9]{64}", case["input_sha256"]):
                raise ValueError("Missing input checksum")
            if set(case["stems"]) != set(stems) or case["numerical_pass"] is not True:
                raise ValueError("Missing stem metrics")
            for m in case["stems"].values():
                score, error = m["legacy_reference_si_sdr_db"], m["max_abs_error"]
                quiet = (
                    number(m["reference_rms"])
                    and m["reference_rms"] <= 1e-5
                    and number(m["rms_error"])
                    and m["rms_error"] <= 1e-8
                    and number(error)
                    and error <= 1e-7
                )
                if (
                    m["samples_per_channel"] != round(case["seconds"] * 44100)
                    or not number(error)
                    or error > 2e-6
                    or not number(m["rms_error"])
                    or not number(m["reference_rms"])
                    or (error != 0 and not quiet and (not number(score) or score < 100))
                ):
                    raise ValueError("Numerical comparison failed")
            if [m["mode"] for m in case["measurements"]] != ["legacy", "bounded"]:
                raise ValueError("Missing isolated measurements")
            for m in case["measurements"]:
                if (
                    m["torch"] != "2.8.0+cu128"
                    or m["cuda_runtime"] != "12.8"
                    or m["device"] != "cuda:0"
                ):
                    raise ValueError("Not the pinned CUDA runtime")
                validate_runtime(m["gpu_runtime"], MIG24)
                for metric in (
                    "seconds_including_io",
                    "process_peak_rss_mib",
                    "cuda_peak_allocated_mib",
                    "cuda_peak_reserved_mib",
                ):
                    if not number(m[metric], positive=True):
                        raise ValueError("Missing positive memory/time measurement")
                if m["cuda_peak_reserved_mib"] < m["cuda_peak_allocated_mib"]:
                    raise ValueError("Invalid CUDA allocator counters")
        if kind == "synthetic":
            short, long = bench["cases"][-2:]
            b60 = short["measurements"][1]["cuda_peak_allocated_mib"]
            b600 = long["measurements"][1]["cuda_peak_allocated_mib"]
            legacy = long["measurements"][0]["cuda_peak_allocated_mib"]
            if b600 > b60 * 1.10 + 32 or b600 >= legacy * 0.5:
                raise ValueError("Bounded GPU memory improvement not demonstrated")


def validate_edges(report, sha, checkpoint_sha):
    """Recompute edge verdicts; a claimed pass or missing boundary is insufficient."""
    lengths = {1}
    for edge in (1024, 4096, 520192, 524288):
        lengths.update((edge - 1, edge, edge + 1))
    expected = {f"boundary_{n}": (n, 2) for n in lengths}
    expected.update(silence=(4097, 2), mono=(4097, 1))
    if (
        report["source_commit"] != sha
        or report["checkpoint_sha256"] != checkpoint_sha
        or report["environment"]["device"] != "cuda"
        or report["passed"] is not True
        or set(report["cases"]) != set(expected)
    ):
        raise ValueError("Incomplete five-stem CUDA edge evidence")
    for name, (length, channels) in expected.items():
        case = report["cases"][name]
        if (
            case["samples"] != length
            or case["channels"] != channels
            or set(case["stems"]) != {"vocals", "piano", "drums", "bass", "other"}
        ):
            raise ValueError("Wrong edge shape/stem set")
        for m in case["stems"].values():
            error, rms, ref, score = (
                m[k]
                for k in (
                    "max_abs_error",
                    "rms_error",
                    "reference_rms",
                    "legacy_reference_si_sdr_db",
                )
            )
            quiet = (
                number(ref)
                and ref <= 1e-5
                and number(rms)
                and rms <= 1e-8
                and number(error)
                and error <= 1e-7
            )
            if (
                m["samples_per_channel"] != length
                or not number(error)
                or error > 2e-6
                or not number(rms)
                or not number(ref)
                or (error != 0 and not quiet and (not number(score) or score < 100))
            ):
                raise ValueError("Five-stem edge numerical failure")
