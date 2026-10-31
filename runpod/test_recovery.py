"""Actual archive/HTTP boundary tests and recovery evidence failure cases."""

import copy
import hashlib
import io
import json
import os
import tarfile
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from test_control import load, worker
from test_training_result import valid_result

recovery = load("recovery_result")
prepare_spec = __import__("importlib.util", fromlist=[""]).spec_from_file_location(
    "prepare", Path(__file__).resolve().parents[1] / "scripts/prepare_musdb_training.py"
)
prepare = __import__("importlib.util", fromlist=[""]).module_from_spec(prepare_spec)
prepare_spec.loader.exec_module(prepare)


def bundle(folder, *, extra=None):
    path = Path(folder) / "checkpoint.tar"
    files = {}
    names = ["last.pt", "metrics.jsonl", "recovery.json"]
    with tarfile.open(path, "w") as tar:
        for name in names + ([extra] if extra else []):
            data = b"checkpoint-test"
            m = tarfile.TarInfo(name)
            m.size = len(data)
            tar.addfile(m, io.BytesIO(data))
            if name in names:
                files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return path, {"bytes": path.stat().st_size, "sha256": recovery.digest(path), "files": files}


def report():
    old = valid_result()["report"]
    r = old["runs"][1]
    hashes = dict.fromkeys(recovery.STATE_KEYS, "b" * 64)
    files = {
        n: {"bytes": 1, "sha256": "c" * 64}
        for n in ("last.pt", "metrics.jsonl", "recovery.json", "best.pt")
    }
    return {
        "status": "passed",
        "report": {
            "kind": "pod-recovery",
            "stage": "resume",
            "commit": "a" * 40,
            "pod_id": "podB",
            "source_pod": "podA",
            "torch": old["torch"],
            "cuda": old["cuda"],
            "gpu": old["gpu"],
            "config": r["config"],
            "assets": old["assets"],
            "cursor": [8, 2, 0],
            "optimizer_steps": [8],
            "weights_finite": True,
            "metrics": r["metrics"]["continuous"],
            "expected_metrics": r["metrics"]["continuous"],
            "expected_last": hashes,
            "expected_best": hashes,
            "actual_last": hashes,
            "actual_best": hashes,
            "inference_finite": True,
            "seconds": 10.0,
            "bundle": {"bytes": 1000, "sha256": "d" * 64, "files": files},
        },
    }


class RecoveryTests(unittest.TestCase):
    def test_real_archive_hash_and_allowlist(self):
        with tempfile.TemporaryDirectory() as f:
            path, desc = bundle(f)
            recovery.validate_bundle(path, desc, complete=False)
            bad = copy.deepcopy(desc)
            bad["files"]["last.pt"]["sha256"] = "0" * 64
            with self.assertRaises(RuntimeError):
                recovery.validate_bundle(path, bad, complete=False)
            path, desc = bundle(f, extra="../../escape")
            with self.assertRaises(RuntimeError):
                recovery.validate_bundle(path, desc, complete=False)
            path, desc = bundle(f, extra="last.pt")
            with self.assertRaises(RuntimeError):
                recovery.validate_bundle(path, desc, complete=False)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(RuntimeError):
                recovery.validate_bundle(path, desc, complete=False)

    def test_bare_pass_same_pod_and_mismatched_states_rejected(self):
        recovery.validate_recovery(report(), "a" * 40, "resume")
        broken = [{"status": "passed"}]
        for k, v in [
            ("source_pod", "podB"),
            ("cursor", [3, 0, 3]),
            ("actual_last", {}),
            ("actual_best", {}),
            ("metrics", []),
            ("optimizer_steps", [3]),
            ("inference_finite", False),
        ]:
            r = copy.deepcopy(report())
            r["report"][k] = v
            broken.append(r)
        for r in broken:
            with self.assertRaises(RuntimeError):
                recovery.validate_recovery(r, "a" * 40, "resume")
        source = copy.deepcopy(report()["report"])
        source["expected_metrics"] = []
        with self.assertRaises(RuntimeError):
            recovery.validate_recovery(report(), "a" * 40, "resume", source)

    def test_authenticated_bounded_once_only_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "restore.tar"
            data = b"checkpoint"
            sha = hashlib.sha256(data).hexdigest()
            server = worker.ThreadingHTTPServer(("127.0.0.1", 0), worker.Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            env = {
                "SPT_TASK": "recovery-resume",
                "SPT_RESTORE_DESCRIPTOR": json.dumps({"bytes": len(data), "sha256": sha}),
            }
            try:
                with (
                    patch.object(worker, "TOKEN", "secret"),
                    patch.object(worker, "RESTORE_FILE", dest),
                    patch.dict(os.environ, env),
                ):
                    url = f"http://127.0.0.1:{server.server_port}/restore"
                    for body, token, status in [
                        (data, "bad", 401),
                        (data + b"x", "secret", 409),
                        (b"x" * len(data), "secret", 400),
                    ]:
                        req = urllib.request.Request(
                            url,
                            data=body,
                            method="PUT",
                            headers={"Authorization": "Bearer " + token},
                        )
                        with self.assertRaises(urllib.error.HTTPError) as err:
                            urllib.request.urlopen(req)
                        self.assertEqual(err.exception.code, status)
                        self.assertFalse(dest.exists())
                    first = urllib.request.Request(
                        url,
                        data=data[:4],
                        method="PUT",
                        headers={"Authorization": "Bearer secret", "X-Offset": "0"},
                    )
                    with urllib.request.urlopen(first) as response:
                        self.assertEqual(json.load(response), {"received": 4})
                    self.assertFalse(dest.exists())
                    with self.assertRaises(urllib.error.HTTPError) as err:
                        urllib.request.urlopen(first)
                    self.assertEqual(err.exception.code, 409)
                    req = urllib.request.Request(
                        url,
                        data=data[4:],
                        method="PUT",
                        headers={"Authorization": "Bearer secret", "X-Offset": "4"},
                    )
                    with urllib.request.urlopen(req) as response:
                        self.assertEqual(
                            json.load(response), {"received": len(data), "sha256": sha}
                        )
                    self.assertEqual(dest.read_bytes(), data)
                    with self.assertRaises(urllib.error.HTTPError) as err:
                        urllib.request.urlopen(req)
                    self.assertEqual(err.exception.code, 409)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_streamed_checkpoint_download_through_actual_http_handler(self):
        transfer = load("transfer")
        with tempfile.TemporaryDirectory() as folder:
            source, descriptor = bundle(folder)
            output = Path(folder) / "downloaded.tar"
            server = worker.ThreadingHTTPServer(("127.0.0.1", 0), worker.Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with (
                    patch.object(worker, "TOKEN", "secret"),
                    patch.object(worker, "EXPORT_FILE", source),
                    patch.object(worker, "RESULT", {"status": "passed"}),
                ):
                    url = f"http://127.0.0.1:{server.server_port}/checkpoint"
                    with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                        transfer.download_bundle(url, "bad", descriptor, output, complete=False)
                    self.assertFalse(output.exists())
                    transfer.download_bundle(url, "secret", descriptor, output, complete=False)
                    self.assertEqual(source.read_bytes(), output.read_bytes())
                    req = urllib.request.Request(
                        url, headers={"Authorization": "Bearer secret", "Range": "bytes=0-99999999"}
                    )
                    with self.assertRaises(urllib.error.HTTPError) as err:
                        urllib.request.urlopen(req)
                    self.assertEqual(err.exception.code, 416)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_full_archive_inventory_rejects_preview_or_unsafe_paths(self):
        names = [f"train/song{i}.stem.mp4" for i in range(100)] + [
            f"test/song{i}.stem.mp4" for i in range(50)
        ]
        self.assertEqual(len(prepare.inventory(names)["train"]), 100)
        for bad in (names[:5], names + [names[0]], names + ["../escape.stem.mp4"]):
            with self.assertRaises(ValueError):
                prepare.inventory(bad)


if __name__ == "__main__":
    unittest.main()
