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


def validate(result, sha, volume, pod, experiment):
    report = result.get("report", {})
    if (
        result.get("status") != "passed"
        or report.get("complete") is not True
        or report.get("kind") != "bounded-memory-v1"
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
    expected_keys = {f"{k}stems-{kind}" for k in (2, 4) for kind in ("synthetic", "real")}
    if set(report["benchmarks"]) != expected_keys:
        raise ValueError("Missing checkpoint/input combinations")
    for key, bench in report["benchmarks"].items():
        model, kind = key.split("-")
        stems = (
            ("vocals", "accompaniment")
            if model == "2stems"
            else ("vocals", "drums", "bass", "other")
        )
        if (
            bench["checkpoint_sha256"] != released[model]["sha256"]
            or bench["source_commit"] != sha
            or bench["requested_device"] != "cuda"
            or bench["numerical_pass"] is not True
            or bench["config"]["instruments"] != list(stems)
            or bench["thresholds"] != {"max_abs_error": 2e-6, "min_si_sdr_db": 100.0}
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
        durations = [1, 12, 60, 600] if kind == "synthetic" else [1, 12, 60]
        if [c["seconds"] for c in bench["cases"]] != durations:
            raise ValueError("Missing requested durations")
        for case in bench["cases"]:
            if not re.fullmatch(r"[a-f0-9]{64}", case["input_sha256"]):
                raise ValueError("Missing input checksum")
            if set(case["stems"]) != set(stems) or case["numerical_pass"] is not True:
                raise ValueError("Missing stem metrics")
            for m in case["stems"].values():
                score, error = m["legacy_reference_si_sdr_db"], m["max_abs_error"]
                if (
                    m["samples_per_channel"] != round(case["seconds"] * 44100)
                    or not number(error)
                    or error > 2e-6
                    or not number(m["rms_error"])
                    or (error != 0 and (not number(score) or score < 100))
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
