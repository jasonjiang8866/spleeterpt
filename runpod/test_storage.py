import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storage import NAME, ensure_volume, safe_mounts
from test_control import ENV, FakeAPI, control, pod, worker


class StorageTests(unittest.TestCase):
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
