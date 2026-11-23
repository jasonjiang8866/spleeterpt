"""Replay measured FP32 control rows in a synthetic profile matrix to test recovery."""

import copy
import gzip
import itertools
import json
import unittest

from precision_result import BASELINE, CONTROL, MODES, OPS, audit


def fixture():
    with gzip.open(BASELINE) as stream:
        study = json.load(stream)
    study["summary"] = {}
    for case in study["cases"].values():
        control = [r for r in case["runs"] if r["profile"] == CONTROL]
        case["runs"] = []
        for repeat in (1, 2):
            arms = [ops for n in range(4) for ops in itertools.combinations(OPS, n)]
            for ops in arms if repeat == 1 else reversed(arms):
                row = copy.deepcopy(control[repeat - 1])
                row["profile"] = "+".join((CONTROL, *ops))
                row["settings"]["fp64_operations"] = list(ops)
                case["runs"].append(row)
                study["summary"][row["profile"]] = {}
    study["screen_passed_profiles"] = []
    return study


class TestPrecisionRecovery(unittest.TestCase):
    def test_control_reproduces_archived_counts(self):
        result = audit(fixture())
        self.assertEqual(result["screen_passed_profiles"], [])
        expected = {"vocals": 0, "piano": 0, "other": 0, "drums": 38, "bass": 146}
        self.assertEqual(
            result["control_tf_failed_samples"], {m: [expected, expected] for m in MODES}
        )
        self.assertTrue(all(result["control_pcm_matches_previous"].values()))
        self.assertTrue(result["tf_reference_matches_previous"])

    def test_nomination_requires_every_comparison_and_matching_repeats(self):
        study = fixture()
        candidate = CONTROL + "+fft64"
        for r in study["cases"]["music_original"]["runs"]:
            if r["profile"] == candidate:
                for stems in r["vs_tf"].values():
                    for metric in stems.values():
                        metric.update(strict_failed_samples=0, max_tolerance_ratio=0.9)
        study["screen_passed_profiles"] = [candidate]
        self.assertEqual(audit(study)["screen_passed_profiles"], [candidate])
        r = next(r for r in study["cases"]["real60"]["runs"] if r["profile"] == candidate)
        r["output_pcm_sha256"]["bounded"]["bass"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "candidates"):
            audit(study)

    def test_incomplete_or_contradictory_measurements_are_rejected(self):
        for failure in ("schedule", "settings", "input", "tf", "verdict", "missing_stem"):
            with self.subTest(failure=failure):
                study = fixture()
                case = study["cases"]["music_original"]
                row = case["runs"][0]
                if failure == "schedule":
                    case["runs"].pop()
                elif failure == "settings":
                    row["settings"]["cudnn_tf32"] = True
                elif failure == "input":
                    case["input_pcm_sha256"] = "b" * 64
                elif failure == "tf":
                    row["vs_tf"]["bounded"]["bass"]["max_tolerance_ratio"] = 0.8
                elif failure == "verdict":
                    row["bounded_vs_full"]["bass"]["max_abs_error"] = 3e-6
                else:
                    del row["output_pcm_sha256"]["full"]["piano"]
                with self.assertRaises(ValueError):
                    audit(study)
