"""Exercise spending rejection and cleanup through the real controller, no cloud calls."""

import copy
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark_budget as budget
import benchmark_control as control
from global_storage import NAME
from gpu_profile import MIG24

ENV = {
    "GITHUB_REPOSITORY": "jasonjiang8866/spleeterpt",
    "GITHUB_REPOSITORY_ID": "123",
    "GITHUB_RUN_ID": "456",
    "GITHUB_RUN_ATTEMPT": "1",
    "GITHUB_SHA": "a" * 40,
    "GITHUB_REF": "refs/heads/main",
    "SPT_CONFIRM_BENCHMARK": "true",
}


class Fake:
    def __init__(self, *, rate=0.69, lost=False):
        self.items, self.calls = [], []
        self.rate, self.lost = rate, lost

    def pods(self):
        return copy.deepcopy(self.items)

    def call(self, path, method="GET", payload=None):
        self.calls.append((path, method, payload))
        if method == "POST":
            self.items.append({"id": "abc", **copy.deepcopy(payload)})
            if self.lost:
                raise RuntimeError("Lost create response")
            return {"id": "abc"}
        if method == "DELETE":
            self.items = [p for p in self.items if "/pods/" + p["id"] != path]
            return
        return {"cost": self.rate, "gpu": {"id": MIG24}}


class Tests(unittest.TestCase):
    def budget_result(self, **kwargs):
        return {
            "data": {
                "myself": {
                    "clientBalance": 9.25,
                    "currentSpendPerHr": 0.01,
                    "dailyCharges": [{"amount": 2.5}],
                    **kwargs,
                }
            }
        }

    def test_budget_private_read_only_query(self):
        with patch.object(budget, "request", return_value=self.budget_result()) as request:
            result = budget.check(SimpleNamespace(key="private-key"))
        self.assertNotIn("9.25", json.dumps(result))
        self.assertNotIn("private-key", json.dumps(result))
        self.assertTrue(result["balance_sufficient"])
        self.assertEqual(request.call_args.kwargs["method"], "POST")
        self.assertNotIn("mutation", request.call_args.kwargs["payload"]["query"])

    def test_budget_missing_low_nonfinite_and_over_month_rejected(self):
        cases = [
            {"clientBalance": 0.99},
            {"clientBalance": True},
            {"clientBalance": float("nan")},
            {"currentSpendPerHr": None},
            {"dailyCharges": None},
            {"dailyCharges": [{"amount": True}]},
            {"dailyCharges": [{"amount": -499.5}]},
            {"dailyCharges": [{"amount": float("inf")}]},
        ]
        for case in cases:
            with (
                self.subTest(case=case),
                patch.object(budget, "request", return_value=self.budget_result(**case)),
                self.assertRaises(RuntimeError),
            ):
                budget.check(SimpleNamespace(key="unused"))

    def run_controller(
        self, api, *, env=None, budget_error=None, polling_error=None, polling_result=None
    ):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            old = Path.cwd()
            os.chdir(temp)
            stack.callback(os.chdir, old)
            stack.enter_context(patch.dict(os.environ, {**ENV, **(env or {})}))
            stack.enter_context(
                patch.object(control, "ensure_global", return_value={"id": "global1", "name": NAME})
            )
            stack.enter_context(patch.object(control, "mig_regions", return_value=["US-PA-1"]))
            stack.enter_context(
                patch.object(control, "check", return_value={}, side_effect=budget_error)
            )
            stack.enter_context(
                patch.object(control, "resolve_image", return_value="image@sha256:" + "a" * 64)
            )
            stack.enter_context(
                patch.object(
                    control,
                    "request",
                    side_effect=(polling_error or control.BenchmarkDeadline("deadline"))
                    if polling_result is None
                    else None,
                    return_value=polling_result,
                )
            )
            stack.enter_context(redirect_stdout(io.StringIO()))
            with self.assertRaises((RuntimeError, control.BenchmarkDeadline)):
                control.launch(api)
            path = Path("runpod-results/provision.json")
            return json.loads(path.read_text()) if path.exists() else None

    def test_paid_gate_and_budget_fail_before_create(self):
        for env, error in [
            ({"SPT_CONFIRM_BENCHMARK": "false"}, None),
            ({"GITHUB_REF": "refs/heads/feature"}, None),
            ({"SPT_BENCHMARK_PROFILE": "unknown"}, None),
            ({}, RuntimeError("budget")),
        ]:
            api = Fake()
            self.run_controller(api, env=env, budget_error=error)
            self.assertEqual(api.calls, [])

    def test_lost_create_response_cleanup_no_retry_or_volume_delete(self):
        api = Fake(lost=True)
        result = self.run_controller(api)
        self.assertEqual([m for _, m, _ in api.calls], ["POST", "DELETE"])
        self.assertEqual(api.items, [])
        self.assertTrue(result["cleanup_confirmed"])
        payload = api.calls[0][2]
        self.assertEqual(
            payload["mounts"], {"global": [{"volumeId": "global1", "path": "/global"}]}
        )
        self.assertNotIn("RUNPOD_API_KEY", payload["env"])

    def test_bad_quote_and_cancellation_cleanup(self):
        for rate in (0.70, True, "nan", 0, -1, 0.69):
            api = Fake(rate=rate)
            result = self.run_controller(api)
            self.assertEqual(sum(method == "POST" for _, method, _ in api.calls), 1)
            self.assertEqual(api.calls[-1][:2], ("/pods/abc", "DELETE"))
            self.assertTrue(result["cleanup_confirmed"])

    def test_failed_acceptance_preserves_export_before_cleanup(self):
        events = []

        def download(url, token, report, path):
            self.assertEqual(report["bundle"], {"bytes": 10})
            path.write_bytes(b"test bytes")
            events.append("download")

        def reject(*args, **kwargs):
            self.assertTrue(Path("runpod-results/result.json").is_file())
            self.assertEqual(
                Path("runpod-results/listening-and-metrics.tar").read_bytes(), b"test bytes"
            )
            events.append("reject")
            raise RuntimeError("Acceptance failure")

        mocks = {
            "benchmark_metrics": SimpleNamespace(timing_summary=lambda *a: {}),
            "benchmark_result": SimpleNamespace(validate=reject),
            "convergence_result": SimpleNamespace(validate=reject),
            "finetune_result": SimpleNamespace(download_export=download),
        }
        api = Fake()
        with patch.dict(sys.modules, mocks):
            meta = self.run_controller(
                api,
                env={"SPT_BENCHMARK_PROFILE": "convergence"},
                polling_result={"status": "failed", "report": {"bundle": {"bytes": 10}}},
            )
        self.assertEqual(events, ["download", "reject"])
        self.assertTrue(meta["cleanup_confirmed"])
        self.assertEqual(api.items, [])

    def test_convergence_profile_pins_snapshot_and_cleanup(self):
        api = Fake()
        result = self.run_controller(api, env={"SPT_BENCHMARK_PROFILE": "convergence"})
        self.assertEqual(result["benchmark_profile"], "convergence")
        self.assertTrue(result["cleanup_confirmed"])
        self.assertEqual(sum(method == "POST" for _, method, _ in api.calls), 1)
        payload = api.calls[0][2]
        self.assertEqual(payload["disk"], 80)
        self.assertEqual(
            payload["env"]["SPT_SNAPSHOT_SHA256"],
            "172c50e5310b22ceab3d4813a4ba1146c682db682c8aebc4c03ad235de376964",
        )
        self.assertEqual(
            payload["mounts"], {"global": [{"volumeId": "global1", "path": "/global"}]}
        )
        self.assertEqual(api.items, [])

    def test_vocal_profile_retains_single_launch_and_cleanup(self):
        api = Fake()
        result = self.run_controller(api, env={"SPT_BENCHMARK_PROFILE": "vocal"})
        self.assertEqual(result["benchmark_profile"], "vocal")
        self.assertTrue(result["cleanup_confirmed"])
        self.assertEqual(sum(method == "POST" for _, method, _ in api.calls), 1)
        self.assertEqual(api.calls[0][2]["env"]["SPT_BENCHMARK_PROFILE"], "vocal")
        self.assertEqual(api.items, [])

    def test_bounded_branch_gate_and_compute_cleanup(self):
        for branch, accepted in [
            ("refs/heads/codex/bounded-memory-inference", True),
            ("refs/heads/main", True),
            ("refs/heads/other", False),
        ]:
            api = Fake()
            result = self.run_controller(
                api, env={"SPT_BENCHMARK_PROFILE": "bounded", "GITHUB_REF": branch}
            )
            self.assertEqual(sum(m == "POST" for _, m, _ in api.calls), int(accepted))
            if accepted:
                self.assertTrue(result["cleanup_confirmed"])
                self.assertEqual(api.calls[0][2]["env"]["SPT_BENCHMARK_PROFILE"], "bounded")
                self.assertNotIn("RUNPOD_API_KEY", api.calls[0][2]["env"])
            self.assertEqual(api.items, [])


if __name__ == "__main__":
    unittest.main()
