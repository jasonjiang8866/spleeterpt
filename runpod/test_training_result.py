"""Acceptance failure cases, including false positives and cleanup after rejection."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_control import ENV, FakeAPI, control, load

validator = load("training_result")


def valid_result():
    root = Path(__file__).resolve().parents[1]
    assets = json.loads((root / "runpod/training-assets.json").read_text())
    rows = []
    for epoch in range(2):
        rows.extend(
            {
                "kind": "train",
                "step": i + 1,
                "epoch": epoch,
                "loss": 1.0,
                "grad_norm": 0.5,
                "update_skipped": False,
            }
            for i in range(epoch * 4, epoch * 4 + 4)
        )
        rows.append({"kind": "validation", "step": epoch * 4 + 4, "epoch": epoch, "loss": 1.0})
    runs = []
    for stems in (2, 4):
        runs.append(
            {
                "stems": stems,
                "config": json.loads((root / f"configs/{stems}stems.json").read_text()),
                "expected_config_matches": True,
                "precision": "fp32",
                "lr": 1e-5,
                "batch_size": 1,
                "epochs": 2,
                "interrupted_cursor": [3, 0, 3],
                "final_cursor": [8, 2, 0],
                "state_equal": dict.fromkeys(validator.STATE_KEYS, True),
                "best_state_equal": True,
                "metrics_equal": True,
                "weights_changed": True,
                "weights_finite": True,
                "inference_finite": True,
                "optimizer_steps": [8],
                "metrics": {"continuous": copy.deepcopy(rows), "resumed": copy.deepcopy(rows)},
                "timings": {
                    k: {"seconds": 1.0, "peak_allocated_bytes": 1024}
                    for k in ("continuous", "interrupted", "resumed")
                },
                "initial_sha256": "a" * 64,
                "last_sha256": "b" * 64,
                "best_sha256": "c" * 64,
            }
        )
    return {
        "status": "passed",
        "report": {
            "kind": "training-preview",
            "commit": "a" * 40,
            "torch": "2.8.0+cu128",
            "cuda": "12.8",
            "gpu": "NVIDIA GeForce RTX 4090",
            "assets": assets,
            "decoded": {
                n: {
                    "samples": 7 * 44100,
                    "wav_sha256": dict.fromkeys(
                        ("mixture", "drums", "bass", "other", "vocals"), "d" * 64
                    ),
                }
                for group in assets.values()
                for n in group
            },
            "runs": runs,
            "seconds": 10.0,
        },
    }


class TrainingResultTests(unittest.TestCase):
    def test_complete_evidence(self):
        validator.validate_training(valid_result(), "a" * 40)

    def test_false_success_is_rejected(self):
        broken = [{"status": "passed"}, {"status": "failed"}]
        for key, val in [
            ("commit", "b" * 40),
            ("cuda", None),
            ("assets", {}),
            ("runs", []),
            ("decoded", {}),
        ]:
            r = valid_result()
            r["report"][key] = val
            broken.append(r)
        for key, val in [
            ("config", {}),
            ("final_cursor", [8, 1, 0]),
            ("interrupted_cursor", [2, 0, 2]),
            ("optimizer_steps", [0, 8]),
            ("state_equal", {}),
            ("best_state_equal", False),
            ("weights_changed", False),
            ("metrics", {"continuous": [], "resumed": []}),
            ("timings", {}),
            ("best_sha256", ""),
        ]:
            r = valid_result()
            r["report"]["runs"][0][key] = val
            broken.append(r)
        for field, value in [
            ("loss", float("nan")),
            ("grad_norm", None),
            ("update_skipped", True),
            ("step", 2),
        ]:
            r = valid_result()
            for rows in r["report"]["runs"][0]["metrics"].values():
                rows[0][field] = value
            broken.append(r)
        for r in broken:
            with self.assertRaises(RuntimeError):
                validator.validate_training(r, "a" * 40)

    def test_rejected_training_still_releases(self):
        api = FakeAPI()

        def reject(*args):
            raise RuntimeError("bad training report")

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict("os.environ", ENV),
            patch.object(control, "Path", side_effect=lambda p: Path(folder) / p),
            patch.object(control, "resolve_image", return_value="image"),
            patch.object(control, "request", return_value={"status": "passed"}),
            patch.dict(sys.modules, {"training_result": SimpleNamespace(validate_training=reject)}),
            self.assertRaisesRegex(RuntimeError, "bad training"),
        ):
            control.launch(api, "training")
        self.assertEqual(api.deleted, ["/pods/abc"])
        self.assertEqual(api.payload["env"]["SPT_TASK"], "training")


if __name__ == "__main__":
    unittest.main()
