"""The resource inspection must never launch, leak credentials or hide blockers."""

import unittest

from benchmark_preflight import ReadOnly, data_plan, observe
from global_storage import NAME as GLOBAL_NAME
from storage import NAME


class Fake:
    def __init__(self, rate=0.69):
        self.calls = []
        self.rate = rate

    def call(self, path, method="GET", payload=None):
        self.calls.append((path, method, payload))
        if path == "/pods":
            return {
                "pods": [{"id": "p1", "env": {"SECRET": "do-not-export"}}],
                "pagination": {"hasNextPage": False},
            }
        if path == "/network-volumes":
            return {
                "networkVolumes": [
                    {
                        "id": "j3aqjpw393",
                        "name": NAME,
                        "size": 80,
                        "type": "STANDARD",
                        "dataCenter": "EU-RO-1",
                    }
                ],
                "pagination": {"hasNextPage": False},
            }
        if path == "/global-volumes":
            return {
                "globalVolumes": [{"id": "global1", "name": GLOBAL_NAME}],
                "pagination": {"hasNextPage": False},
            }
        if path.startswith("/catalog/gpus/"):
            return {"price": {"secure": self.rate}, "dataCenters": []}
        raise AssertionError(path)


class Tests(unittest.TestCase):
    def test_get_only_and_no_secrets(self):
        api = Fake()
        report = observe(api)
        self.assertTrue(all(m == "GET" and p is None for _, m, p in api.calls))
        self.assertNotIn("do-not-export", str(report))
        self.assertEqual(report["account_pod_count"], 1)
        self.assertFalse(report["ready_for_paid_run"])
        with self.assertRaises(RuntimeError):
            ReadOnly(api).call("/pods", "POST", {})

    def test_missing_or_excessive_price_is_not_accepted(self):
        for rate in (None, True, float("nan"), float("inf"), -1, 0, 0.70):
            with self.subTest(rate=rate):
                self.assertFalse(observe(Fake(rate))["gpu"]["within_proposed_rate_cap"])

    def test_missing_regional_volume_does_not_hide_global_inventory(self):
        class MissingSource(Fake):
            def call(self, path, method="GET", payload=None):
                result = super().call(path, method, payload)
                if path == "/network-volumes":
                    result["networkVolumes"] = []
                return result

        report = observe(MissingSource())
        self.assertIsNone(report["regional_volume"])
        self.assertEqual(report["global_volume"]["id"], "global1")
        self.assertFalse(report["ready_for_paid_run"])

    def test_pinned_split(self):
        plan = data_plan()
        self.assertEqual(len(plan["track_ids"]), 10)
        self.assertFalse(plan["test_pool_used"])
        self.assertEqual(plan["expected_audio_files"], 50)

    def test_vocal_inventory_does_not_label_four_stem_baseline_as_native_two(self):
        plan = data_plan(vocal=True)
        self.assertNotEqual(plan["spleeter_initial_sha256"], data_plan()["spleeter_initial_sha256"])
        self.assertEqual(plan["track_ids"], data_plan()["track_ids"])
        report = observe(Fake(), vocal=True)
        self.assertIn("Kim Mel-Band", str(report["pending"]))
        self.assertNotIn("HTDemucs", str(report["pending"]))


if __name__ == "__main__":
    unittest.main()
