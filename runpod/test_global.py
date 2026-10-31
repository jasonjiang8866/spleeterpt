"""Cloud-free tests of integrity, billing bounds, placement and retained-volume cleanup."""

import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import global_inputs as inputs
import global_storage as gs
from storage import NAME, safe_mounts
from test_control import ENV, FakeAPI, control, pod


class GlobalTests(unittest.TestCase):
    def test_global_ids_are_opaque_not_a_documentation_example_prefix(self):
        for value in ("gsb_unit", "550e8400-e29b-41d4-a716-446655440000", "abc123"):
            self.assertTrue(gs.valid_id(value))
        for value in ("", "../other", "two words", "id\n", None):
            self.assertFalse(gs.valid_id(value))

    def test_cli_global_export_dispatches_compute_instead_of_cleanup(self):
        with (
            patch.dict(os.environ, ENV),
            patch("sys.argv", ["control.py", "global-export"]),
            patch.object(control, "Runpod") as api,
            patch.object(control, "launch") as launch,
            patch.object(control, "cleanup") as cleanup,
        ):
            control.main()
        launch.assert_called_once_with(api.return_value, "global-export")
        cleanup.assert_not_called()

    def test_global_gpu_uses_verified_snapshot_without_regional_mount(self):
        api = FakeAPI()
        api.rate = ".70"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "global-input").mkdir()
            (root / "global-input/result.json").write_text(
                json.dumps({"report": {"pod_id": "source", "snapshot_sha256": "a" * 64}})
            )
            with (
                patch.dict(
                    os.environ, {**ENV, "SPT_GLOBAL_BN": "1", "SPT_EXPERIMENT": inputs.EXPERIMENT}
                ),
                patch.object(control, "Path", side_effect=lambda p: root / p),
                patch.object(control, "resolve_image", return_value="digest"),
                patch(
                    "storage.ensure_volume",
                    side_effect=AssertionError("must not pin source region"),
                ),
                patch.object(gs, "ensure_global", return_value={"id": "gsb_unit", "name": gs.NAME}),
                patch.object(gs, "mig_regions", return_value=["US-PA-1"]),
                patch.object(inputs, "validate_report") as validate,
                self.assertRaisesRegex(RuntimeError, "rate"),
            ):
                control.launch(api, "bn")
            validate.assert_called_once()
        self.assertEqual(
            api.payload["mounts"], {"global": [{"volumeId": "gsb_unit", "path": "/global"}]}
        )
        self.assertEqual(api.payload["dataCenterIds"], ["US-PA-1"])
        self.assertEqual(api.deleted, ["/pods/abc"])

    def test_global_identity_and_no_automatic_create_retry(self):
        api = FakeAPI()
        v = {"id": "gsb_unit", "name": gs.NAME}
        for vs in ([v], [v, v], [{**v, "id": "bad/id"}]):
            with patch.object(
                api,
                "call",
                return_value={"globalVolumes": vs, "pagination": {"hasNextPage": False}},
            ):
                if vs == [v]:
                    self.assertEqual(gs.ensure_global(api), v)
                else:
                    with self.assertRaises(RuntimeError):
                        gs.ensure_global(api)
        with patch.object(
            api,
            "call",
            side_effect=[
                {"globalVolumes": [], "pagination": {"hasNextPage": False}},
                RuntimeError("Lost POST"),
            ],
        ) as call:
            with self.assertRaisesRegex(RuntimeError, "Lost POST"):
                gs.ensure_global(api, create=True)
            self.assertEqual(call.call_count, 2)
            self.assertEqual(call.call_args.args, ("/global-volumes", "POST", {"name": gs.NAME}))

    def test_cpu_bound_and_global_mig_do_not_pin_source_region(self):
        api = FakeAPI()
        cpu = {
            "id": "cpu5c",
            "price": {"securePerVcpu": 0.03},
            "vcpu": {"min": 2, "max": 32},
            "dataCenters": [{"id": "EU-RO-1", "availability": "LOW"}],
        }
        with patch.object(api, "call", return_value={"cpus": [cpu]}) as call:
            self.assertEqual(gs.cpu_placement(api, "EU-RO-1"), {"id": "cpu5c", "vcpuCount": 2})
            self.assertEqual(len(call.call_args.args), 1)
            with self.assertRaises(RuntimeError):
                gs.cpu_placement(api, "US-PA-1")
            cpu["price"]["securePerVcpu"] = 0.11
            with self.assertRaises(RuntimeError):
                gs.cpu_placement(api, "EU-RO-1")
        with patch.object(
            api,
            "call",
            return_value={
                "price": {"secure": 0.69},
                "dataCenters": [{"id": "US-PA-1", "availability": "MEDIUM"}],
            },
        ):
            self.assertEqual(gs.mig_regions(api, "gpu"), ["US-PA-1"])

    def test_cleanup_retains_both_declared_volumes_and_rejects_others(self):
        p = pod()
        p["env"].update(
            SPT_TASK="global-export",
            SPT_GLOBAL_VOLUME_ID="gsb_unit",
            SPT_GLOBAL_VOLUME_NAME=gs.NAME,
            SPT_VOLUME_ID=gs.SOURCE_ID,
            SPT_VOLUME_NAME=NAME,
        )
        p["mounts"] = {
            "network": [{"volumeId": gs.SOURCE_ID, "path": "/data"}],
            "global": [{"volumeId": "gsb_unit", "path": "/global"}],
        }
        for evaluation in (False, True):
            q = copy.deepcopy(p)
            if evaluation:
                q["env"].update(SPT_TASK="bn", SPT_VOLUME_ID="gsb_unit")
                del q["mounts"]["network"]
            self.assertTrue(safe_mounts(q))
            api = FakeAPI()
            api.items = [q]
            with patch.dict(os.environ, ENV):
                control.cleanup(api)
            self.assertEqual(api.deleted, ["/pods/pod456"])
            q["mounts"]["global"][0]["volumeId"] = "gsb_other"
            self.assertFalse(safe_mounts(q))

    def test_cpu_overprice_and_lost_response_cleanup(self):
        volume = {"id": gs.SOURCE_ID, "name": NAME, "dataCenter": "EU-RO-1"}
        for lost in (False, True):
            api = FakeAPI()
            api.rate, api.lose_response = ".21", lost
            with (
                tempfile.TemporaryDirectory() as folder,
                patch.dict(os.environ, ENV),
                patch.object(control, "Path", side_effect=lambda p: Path(folder) / p),
                patch.object(control, "resolve_image", return_value="digest"),
                patch("storage.ensure_volume", return_value=volume),
                patch.object(gs, "cpu_placement", return_value={"id": "cpu5c", "vcpuCount": 2}),
                patch.object(gs, "ensure_global", return_value={"id": "gsb_unit", "name": gs.NAME}),
                self.assertRaises(RuntimeError),
            ):
                control.launch(api, "global-export")
            self.assertEqual(api.creates, 1)
            self.assertEqual(api.deleted, ["/pods/abc"])
            self.assertNotIn("gpu", api.payload)
            self.assertNotIn("RUNPOD_API_KEY", api.payload["env"])
            self.assertEqual(api.payload["cpu"]["vcpuCount"], 2)

    def test_copy_integrity_resume_and_path_containment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target = root / "source", root / "target"
            source.mkdir()
            (source / "audio").write_bytes(b"correct")
            spec = {
                "files": {"audio": {"bytes": 7, "sha256": hashlib.sha256(b"correct").hexdigest()}}
            }
            with redirect_stdout(io.StringIO()):
                inputs.copy_inputs(source, target, spec)
                inputs.copy_inputs(source, target, spec)
            self.assertEqual((target / "audio").read_bytes(), b"correct")
            (target / "audio").write_bytes(b"partial")
            (source / "audio").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                inputs.copy_inputs(source, target, spec)
            for name in ("../escape", "/absolute"):
                with self.assertRaises(ValueError):
                    inputs.checked_path(source, name)
            (source / "link").symlink_to(root)
            with self.assertRaises(ValueError):
                inputs.checked_path(source, "link/file")

    def test_snapshot_inventory_is_derived_from_pinned_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "docs/results").mkdir(parents=True)
            tracks = [
                {
                    "id": f"train/song-{i}.stem.mp4",
                    "samples": 10,
                    "source_sha256": "a" * 64,
                    "files": {
                        s: {"bytes": 1, "sha256": "b" * 64}
                        for s in ("mixture", "vocals", "drums", "bass", "other")
                    },
                }
                for i in range(100)
            ]
            pinned = {
                "archive_sha256": "c" * 64,
                "tracks": tracks,
                "split_track_ids": {"validation": [f"train/song-{i}" for i in range(10)]},
            }
            provenance = json.dumps(
                {"archive_sha256": "c" * 64, "test_audio_extracted": False, "tracks": tracks}
            )
            ev = {
                "resume": {
                    "dataset": {
                        "provenance_sha256": hashlib.sha256(provenance.encode()).hexdigest()
                    },
                    "backup": {
                        "generation": "step-000000090-" + "a" * 32,
                        "files": {"last.pt": {"bytes": 1, "sha256": "d" * 64}},
                    },
                }
            }
            (root / "docs/results/musdb18-training-provenance.json").write_text(json.dumps(pinned))
            with (
                patch.object(inputs, "ROOT", root),
                patch.object(inputs, "evidence", return_value=ev),
            ):
                spec = inputs.manifest(provenance)
                self.assertEqual(spec["verified_audio_files"], 50)
                self.assertEqual(len(spec["files"]), 54)
                self.assertFalse(any("song-99/" in name for name in spec["files"]))
                with self.assertRaisesRegex(ValueError, "checksum"):
                    inputs.manifest(provenance + " ")
                with (
                    patch.object(inputs, "MAX_BYTES", 1),
                    self.assertRaisesRegex(ValueError, "8 GiB"),
                ):
                    inputs.manifest(provenance)
