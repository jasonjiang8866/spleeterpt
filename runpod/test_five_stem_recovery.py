import copy
import gzip
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from five_stem_recovery import verify
from gpu_profile import MIG24
from precision_result import CONTROL, ROOT, compare_cli_to_screen
from test_five_stem_result import fixture


def prepare(root, cli=None):
    result = fixture()
    if cli is not None:
        result["report"]["cuda_cli"] = cli
    members = {"report.json": result["report"]}
    if cli is not None:
        members["cuda-cli.json"] = cli
    archive = root / "listening-and-metrics.tar"
    files = {}
    with tarfile.open(archive, "w") as tar:
        for name, data in members.items():
            raw = json.dumps(data).encode()
            files[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    result["report"]["bundle"] = {
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": files,
    }
    (root / "result.json").write_text(json.dumps(result))
    (root / "provision.json").write_text(
        json.dumps(
            {
                "cleanup_confirmed": True,
                **(
                    {"cli_acceptance": True, "cuda_cli_profile": "fft64"} if cli is not None else {}
                ),
                "benchmark_profile": "five-stem",
                "commit": "a" * 40,
                "global_volume_id": "global1",
                "pod_id": "pod1",
                "gpu_id": MIG24,
            }
        )
    )


class Tests(unittest.TestCase):
    def test_reopens_export_without_compute(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepare(root)
            verify(root)
            self.assertTrue(
                json.loads((root / "recovery.json").read_text())["all_members_verified"]
            )

    def test_rejected_metrics_can_be_recovered_but_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepare(root)
            result = json.loads((root / "result.json").read_text())
            result["status"] = "failed"
            (root / "result.json").write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                verify(root)
            verify(root, allow_rejected=True)
            recovery = json.loads((root / "recovery.json").read_text())
            self.assertTrue(recovery["all_members_verified"])
            self.assertFalse(recovery["gpu_acceptance_revalidated"])
            self.assertEqual(recovery["acceptance_status"], "rejected")
            result["report"]["commit"] = "b" * 40
            (root / "result.json").write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                verify(root, allow_rejected=True)

    def test_screen_replay_from_a_rejected_partial_cli_archive(self):
        with gzip.open(ROOT / "docs/results/cuda-precision-20261124/cuda-precision.json.gz") as f:
            screen = json.load(f)
        music = screen["cases"]["music_original"]
        measured = next(r for r in music["runs"] if r["profile"] == CONTROL + "+fft64")
        # Synthesize only the public CLI report envelope; the hashes are from the real screen.
        original = {
            "input_pcm_sha256": music["input_pcm_sha256"],
            "reference_sha256": music["reference_sha256"],
            "output_pcm_sha256": copy.deepcopy(measured["output_pcm_sha256"]),
        }
        sha = "a" * 40
        probe = {
            "cli_profile": "fft64",
            "source_commit": sha,
            "checkpoint_sha256": screen["checkpoint_sha256"],
            "calibration": {"music_original": original},
        }
        match = compare_cli_to_screen(probe, sha)
        self.assertEqual(match["tf_reference_differs"], [])
        self.assertEqual(match["cli_output_differs"], {"full": [], "bounded": []})
        self.assertTrue(match["same_checkpoint"] and match["same_input_pcm"])
        original["output_pcm_sha256"]["bounded"]["piano"] = "0" * 64
        original["reference_sha256"]["drums"] = "1" * 64
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare(root, probe)
            verify(root, allow_rejected=True)
            recovered = json.loads((root / "recovery.json").read_text())
            self.assertEqual(recovered["acceptance_status"], "rejected")
            replay = recovered["cuda_cli_screen_replay"]
            self.assertEqual(replay["status"], "compared")
            self.assertEqual(replay["tf_reference_differs"], ["drums"])
            self.assertEqual(replay["cli_output_differs"], {"full": [], "bounded": ["piano"]})
        with self.assertRaisesRegex(ValueError, "source commit"):
            compare_cli_to_screen(probe, "b" * 40)
        del original["output_pcm_sha256"]["full"]["bass"]
        with self.assertRaisesRegex(ValueError, "PCM hashes"):
            compare_cli_to_screen(probe, sha)
        original["input_pcm_sha256"] = "0" * 64
        mismatch = compare_cli_to_screen(probe, sha)
        self.assertEqual(mismatch["status"], "different_input_or_checkpoint")
        self.assertNotIn("cli_output_differs", mismatch)
        probe["calibration"].clear()
        self.assertEqual(compare_cli_to_screen(probe, sha), {"status": "pending_music_original"})

    def test_rejects_missing_cleanup_or_corruption(self):
        for corrupt in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                prepare(root)
                if corrupt:
                    (root / "listening-and-metrics.tar").write_bytes(b"broken")
                else:
                    meta = json.loads((root / "provision.json").read_text())
                    meta["cleanup_confirmed"] = False
                    (root / "provision.json").write_text(json.dumps(meta))
                with self.assertRaises((ValueError, RuntimeError)):
                    verify(root)
                self.assertFalse((root / "recovery.json").exists())


if __name__ == "__main__":
    unittest.main()
