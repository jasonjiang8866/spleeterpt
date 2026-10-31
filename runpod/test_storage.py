import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storage import NAME, ensure_volume, safe_mounts
from test_control import ENV, FakeAPI, control, pod, worker


class StorageTests(unittest.TestCase):
    def test_only_fullsong_evaluation_gets_longer_lease_with_same_price_cleanup_guard(self):
        volume = {
            "id": "j3aqjpw393",
            "name": NAME,
            "size": 80,
            "type": "STANDARD",
            "dataCenter": "EU-RO-1",
        }
        for task, seconds in (("smoke", 1800), ("finetune", 3600), ("bn", 1800)):
            api = FakeAPI()
            api.rate = "1.01"
            with (
                tempfile.TemporaryDirectory() as folder,
                patch.dict("os.environ", {**ENV, "SPT_EXPERIMENT": "41189465452-1"}),
                patch.object(control, "Path", side_effect=lambda p: Path(folder) / p),
                patch.object(control, "resolve_image", return_value="image"),
                patch.object(control.time, "time", return_value=100),
                patch("storage.ensure_volume", return_value=volume),
                self.assertRaisesRegex(RuntimeError, "rate"),
            ):
                control.launch(api, task)
            self.assertEqual(int(api.payload["env"]["SPT_EXPIRY"]), 100 + seconds)
            self.assertEqual(api.creates, 1)
            self.assertEqual(api.deleted, ["/pods/abc"])

    def test_volume_selection_fails_closed_and_never_deletes(self):
        v = {"id": "vol123", "name": NAME, "size": 80, "type": "STANDARD", "dataCenter": "US-KS-2"}
        api = FakeAPI()
        with patch.object(
            api, "call", return_value={"networkVolumes": [v], "pagination": {"hasNextPage": False}}
        ) as call:
            self.assertEqual(ensure_volume(api), v)
            self.assertEqual(call.call_count, 1)
        for item in ([v, v], [{**v, "size": 500}]):
            with (
                patch.object(
                    api,
                    "call",
                    return_value={"networkVolumes": item, "pagination": {"hasNextPage": False}},
                ),
                self.assertRaises(RuntimeError),
            ):
                ensure_volume(api, create=True)

    def test_only_declared_network_mount_survives_compute_cleanup(self):
        p = pod()
        p["mounts"] = {"network": [{"volumeId": "vol123", "path": "/data"}]}
        p["env"].update(SPT_VOLUME_ID="vol123", SPT_VOLUME_NAME=NAME)
        self.assertTrue(safe_mounts(p))
        api = FakeAPI()
        api.items = [p]
        with patch.dict("os.environ", ENV):
            control.cleanup(api)
        self.assertEqual(api.deleted, ["/pods/pod456"])
        for mounts in (
            {"persistent": {"size": 80, "path": "/data"}},
            {"network": [{"volumeId": "other", "path": "/data"}]},
            {"network": [{"volumeId": "vol123", "path": "/workspace"}]},
        ):
            bad = copy.deepcopy(p)
            bad["mounts"] = mounts
            self.assertFalse(safe_mounts(bad))
            with self.assertRaises(RuntimeError):
                control.owned(bad, ENV["GITHUB_REPOSITORY"], "123")


class ProgressTests(unittest.TestCase):
    def test_only_complete_allowlisted_progress_lines_are_exposed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "log"
            path.write_text("private diagnostics\nProgress: archive chunks 10/70\nsecret=hidden\n")
            self.assertEqual(worker.progress(path), "Progress: archive chunks 10/70")
            path.write_text("Progress: archive chunks 10/70 secret=hidden\n")
            self.assertIsNone(worker.progress(path))
            path.write_text("Durable checkpoint committed: step 10\n" + "x" * 70000)
            self.assertIsNone(worker.progress(path))


class PersistentAcceptanceTests(unittest.TestCase):
    def test_real_step_ten_report_accepts_tuple_and_json_config(self):
        import json

        from persistent_result import validate_persistent

        r = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "docs/results/persistent-pilot-41189465452.json"
            ).read_text()
        )["interrupted_training"]
        for in_memory in (False, True):
            report = copy.deepcopy(r)
            if in_memory:
                for key in ("instruments", "filters"):
                    report["config"][key] = tuple(report["config"][key])
            validate_persistent(
                {"status": "passed", "report": report},
                r["commit"],
                "persistent-train",
                r["volume_id"],
                r["pod_id"],
            )
            report["config"]["encoder_activation"] = "relu"
            with self.assertRaisesRegex(RuntimeError, "wrong config"):
                validate_persistent(
                    {"status": "passed", "report": report},
                    r["commit"],
                    "persistent-train",
                    r["volume_id"],
                    r["pod_id"],
                )
