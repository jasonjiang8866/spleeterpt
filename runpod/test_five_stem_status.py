import unittest
from unittest.mock import Mock, patch

from five_stem_status import observe
from gpu_profile import MIG24


class Tests(unittest.TestCase):
    def test_only_allowlisted_status_leaves_runner(self):
        repo = "jasonjiang8866/spleeterpt"
        pod = {
            "id": "pod1",
            "name": "spt-1436489970-123-1-9999999999",
            "cost": 0.69,
            "env": {
                "SPT_MANAGED_BY": repo,
                "SPT_BENCHMARK_PROFILE": "five-stem",
                "SPT_COMMIT": "a" * 40,
                "SPT_GPU_ID": MIG24,
                "SPT_RESULT_TOKEN": "private-token",
            },
        }
        api = Mock()
        api.call.return_value = {"pods": [pod], "pagination": {"hasNextPage": False}}
        with patch("five_stem_status.request") as request:
            request.return_value = {
                "status": "running",
                "phase": "benchmark: command 3/3",
                "progress": "private-token",
                "report": {"untrusted": "private-token"},
            }
            report = observe(api, repo, "1436489970")
            self.assertEqual(report["pods"][0]["phase"], "benchmark: command 3/3")
            self.assertNotIn("private-token", str(report))
            self.assertEqual(api.call.call_args.args, ("/pods",))
            self.assertFalse(api.call.call_args.kwargs)
            request.side_effect = RuntimeError("private-token")
            report = observe(api, repo, "1436489970")
            self.assertEqual(report["pods"][0]["worker_status"], "unreachable")
            self.assertNotIn("private-token", str(report))


if __name__ == "__main__":
    unittest.main()
