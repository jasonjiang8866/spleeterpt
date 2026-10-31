"""Cloud-free bounds and complete-training-snapshot checks."""

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import global_inputs as inputs
import global_storage as gs
from gpu_profile import MIG24, profile
from test_control import ENV, FakeAPI, control
from storage import safe_mounts


class FrozenTests(unittest.TestCase):
    def test_full_training_snapshot_preserves_split_bytes_and_excludes_test_pool(self):
        pinned = json.loads(
            (inputs.ROOT / "docs/results/musdb18-training-provenance.json").read_text()
        )
        # The repository provenance was prepared at a different filesystem root.
        # Reconstruct this fixture's exact persistent /data manifest bytes.
        for split, tracks in pinned["split_track_ids"].items():
            data = "".join(
                json.dumps({"track": "/data/musdb18/train-pool/" + t.removeprefix("train/")}) + "\n"
                for t in tracks
            ).encode()
            pinned["manifests"][split + ".jsonl"] = hashlib.sha256(data).hexdigest()
        raw = json.dumps(pinned)
        ev = copy.deepcopy(inputs.evidence())
        ev["resume"]["dataset"]["provenance_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
        with patch.object(inputs, "evidence", return_value=ev):
            spec = inputs.manifest(raw, training=True)
            self.assertEqual(len(spec["files"]), 503)
            self.assertEqual(spec["verified_audio_files"], 500)
            self.assertLess(spec["bytes"], 40 * 1024**3)
            self.assertEqual(spec["kind"], inputs.TRAIN_PREFIX)
            self.assertFalse(any("/test/" in n or "experiments/" in n for n in spec["files"]))
            for split in ("train", "validation"):
                data = inputs.generated(raw, training=True)["musdb18/manifests/" + split + ".jsonl"]
                self.assertEqual(
                    hashlib.sha256(data).hexdigest(), pinned["manifests"][split + ".jsonl"]
                )
            with self.assertRaises(ValueError):
                inputs.manifest(raw + " ", training=True)

    def test_frozen_launch_caps_price_uses_global_only_and_releases_on_failure(self):
        self.assertEqual(profile("frozen"), (MIG24, 0.69))
        api = FakeAPI()
        api.rate = ".70"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "global-input").mkdir()
            (root / "global-input/result.json").write_text(
                json.dumps({"report": {"pod_id": "source", "snapshot_sha256": "a" * 64}})
            )
            with (
                patch.dict(os.environ, ENV, clear=True),
                patch.object(control, "Path", side_effect=lambda p: root / p),
                patch.object(control, "resolve_image", return_value="digest"),
                patch("storage.ensure_volume", side_effect=AssertionError("regional pin")),
                patch.object(gs, "ensure_global", return_value={"id": "unit", "name": gs.NAME}),
                patch.object(gs, "mig_regions", return_value=["US-PA-1"]),
                patch.object(inputs, "validate_report") as check,
                self.assertRaisesRegex(RuntimeError, "rate"),
            ):
                control.launch(api, "frozen")
            self.assertTrue(check.call_args.kwargs["training"])
        self.assertEqual(api.payload["disk"], 80)
        self.assertEqual(api.payload["gpu"]["id"], MIG24)
        self.assertEqual(
            api.payload["mounts"], {"global": [{"volumeId": "unit", "path": "/global"}]}
        )
        self.assertTrue(safe_mounts(api.payload))
        self.assertEqual(api.deleted, ["/pods/abc"])
