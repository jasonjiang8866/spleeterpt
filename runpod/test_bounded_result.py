"""Reject incomplete/CPU/falsely-passing benchmark reports before acceptance."""

import copy
import json
import unittest

from bounded_result import validate
from gpu_profile import MIG24
from quality import ROOT


def fixture():
    runtime = {
        "requested_gpu": MIG24,
        "name": MIG24,
        "device_count": 1,
        "capability": [12, 0],
        "total_memory_bytes": 24 * 1024**3,
    }
    report = {
        "kind": "bounded-memory-v1",
        "commit": "a" * 40,
        "complete": True,
        "global_volume_id": "global1",
        "pod_id": "pod1",
        "experiment": "1-1",
        "gpu_runtime": runtime,
        "cuda_tests": {"returncode": 0, "output": "3 passed"},
        "real_input": {"test_pool_used": False, "sha256": "b" * 64},
        "benchmarks": {},
    }
    for n in (2, 4):
        base = json.loads((ROOT / f"docs/results/bounded-{n}stems-cpu.json").read_text())
        for kind in ("synthetic", "real"):
            bench = copy.deepcopy(base)
            bench.update(source_commit="a" * 40, requested_device="cuda", input_sha256="b" * 64)
            bench["thresholds"].update(
                near_silent_reference_rms=1e-5,
                near_silent_rms_error=1e-8,
                near_silent_max_abs_error=1e-7,
            )
            if kind == "real":
                bench["cases"] = bench["cases"][:3]
            else:
                bench["cases"][-1]["seconds"] = 600
            for case in bench["cases"]:
                for m in case["stems"].values():
                    m["samples_per_channel"] = int(case["seconds"] * 44100)
                    m["reference_rms"] = 0.1
                for m in case["measurements"]:
                    peak = 1000 if m["mode"] == "bounded" else case["seconds"] * 40
                    m.update(
                        torch="2.8.0+cu128",
                        cuda_runtime="12.8",
                        device="cuda:0",
                        gpu_runtime=runtime,
                        cuda_peak_allocated_mib=peak,
                        cuda_peak_reserved_mib=peak + 100,
                    )
            report["benchmarks"][f"{n}stems-{kind}"] = bench
    return {"status": "passed", "report": report}


class Tests(unittest.TestCase):
    def check(self, result):
        validate(result, "a" * 40, "global1", "pod1", "1-1")

    def test_complete_fixture(self):
        self.check(fixture())

    def test_near_silent_requires_small_absolute_errors(self):
        result = fixture()
        metric = result["report"]["benchmarks"]["4stems-real"]["cases"][1]["stems"]["bass"]
        metric.update(
            reference_rms=3e-6,
            rms_error=6.5e-11,
            max_abs_error=2.5e-9,
            legacy_reference_si_sdr_db=94,
        )
        self.check(result)
        for changed in (
            {"rms_error": 2e-8},
            {"max_abs_error": 2e-7},
            {"reference_rms": 0.1},
            {"reference_rms": float("nan")},
        ):
            bad = copy.deepcopy(result)
            bad["report"]["benchmarks"]["4stems-real"]["cases"][1]["stems"]["bass"].update(changed)
            with self.assertRaises(ValueError):
                self.check(bad)

    def test_bad_evidence_rejected_even_with_pass_flag(self):
        for mutation in (
            lambda r: r.update(complete=False),
            lambda r: r.update(commit="c" * 40),
            lambda r: r["cuda_tests"].update(output="3 skipped"),
            lambda r: r["benchmarks"].pop("4stems-real"),
            lambda r: r["benchmarks"]["2stems-synthetic"]["config"].update(hop_length=512),
            lambda r: r["benchmarks"]["2stems-synthetic"]["cases"][0]["stems"]["vocals"].update(
                max_abs_error=0.01
            ),
            lambda r: r["benchmarks"]["2stems-synthetic"]["cases"][0]["measurements"][0].update(
                device="cpu"
            ),
            lambda r: r["benchmarks"]["2stems-synthetic"]["cases"][-1]["measurements"][1].update(
                cuda_peak_allocated_mib=18000, cuda_peak_reserved_mib=20000
            ),
        ):
            result = fixture()
            mutation(result["report"])
            with self.assertRaises((ValueError, KeyError)):
                self.check(result)


if __name__ == "__main__":
    unittest.main()
