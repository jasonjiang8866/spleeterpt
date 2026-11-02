import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from five_stem_recovery import verify
from gpu_profile import MIG24
from test_five_stem_result import fixture


def prepare(root):
    result = fixture()
    raw = json.dumps(result["report"]).encode()
    archive = root / "listening-and-metrics.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("report.json")
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    result["report"]["bundle"] = {
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": {"report.json": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}},
    }
    (root / "result.json").write_text(json.dumps(result))
    (root / "provision.json").write_text(
        json.dumps(
            {
                "cleanup_confirmed": True,
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
