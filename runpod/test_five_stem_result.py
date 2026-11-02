"""Five-stem GPU reports must prove identity, every stem and real CUDA edges."""

import copy
import json
import unittest

from bounded_result import validate
from quality import ROOT
from test_bounded_result import fixture as four_fixture


def fixture():
    result = four_fixture()
    report = result["report"]
    pinned = json.loads((ROOT / "scripts/five_stem_source.json").read_text())
    report["kind"] = "five-stem-bounded-v1"
    report["conversion"] = {"source": pinned, "checkpoint_sha256": pinned["converted_sha256"]}
    benchmarks = {}
    for kind in ("synthetic", "real"):
        bench = report["benchmarks"]["4stems-" + kind]
        bench["checkpoint_sha256"] = pinned["converted_sha256"]
        bench["config"]["instruments"] = pinned["instruments"]
        for case in bench["cases"]:
            case["stems"]["piano"] = copy.deepcopy(case["stems"]["other"])
        benchmarks["5stems-" + kind] = bench
    report["piano_source"] = json.loads((ROOT / "scripts/piano_source.json").read_text())
    report["piano_inputs"] = {}
    for start in (0, 15, 30):
        key = f"5stems-piano-{start}"
        bench = copy.deepcopy(benchmarks["5stems-real"])
        bench["cases"] = [bench["cases"][1]]
        benchmarks[key] = bench
        report["piano_inputs"][key] = {
            "sha256": "b" * 64,
            "start_seconds": start,
            "samples": 12 * 44100,
        }
    report["benchmarks"] = benchmarks
    edges = {
        "source_commit": "a" * 40,
        "checkpoint_sha256": pinned["converted_sha256"],
        "environment": {"device": "cuda"},
        "passed": True,
        "cases": {},
    }
    lengths = {1}
    for edge in (1024, 4096, 520192, 524288):
        lengths.update((edge - 1, edge, edge + 1))
    shapes = {f"boundary_{n}": (n, 2) for n in lengths}
    shapes.update(silence=(4097, 2), mono=(4097, 1))
    for name, (length, channels) in shapes.items():
        metrics = copy.deepcopy(benchmarks["5stems-real"]["cases"][0]["stems"])
        for m in metrics.values():
            m["samples_per_channel"] = length
        edges["cases"][name] = {"samples": length, "channels": channels, "stems": metrics}
    report["edges"] = edges
    return result


class Tests(unittest.TestCase):
    def check(self, result):
        validate(result, "a" * 40, "global1", "pod1", "1-1", five_stem=True)

    def test_complete(self):
        self.check(fixture())

    def test_reject_claimed_pass_with_missing_or_wrong_evidence(self):
        for mutate in (
            lambda r: r["conversion"].update(checkpoint_sha256="0" * 64),
            lambda r: r["edges"]["cases"].pop("mono"),
            lambda r: r["edges"]["environment"].update(device="cpu"),
            lambda r: r["edges"]["cases"]["silence"]["stems"].pop("piano"),
            lambda r: r["edges"]["cases"]["boundary_1"]["stems"]["piano"].update(
                max_abs_error=0.01
            ),
            lambda r: r["benchmarks"]["5stems-real"]["cases"][0]["stems"].pop("piano"),
        ):
            result = fixture()
            mutate(result["report"])
            with self.assertRaises(ValueError):
                self.check(result)


if __name__ == "__main__":
    unittest.main()
