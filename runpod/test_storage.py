import copy
import unittest
from unittest.mock import patch

from storage import NAME, ensure_volume, safe_mounts
from test_control import ENV, FakeAPI, control, pod


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
