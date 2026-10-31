"""Offline failure-path tests. Never use cloud credentials or rent a GPU."""

import copy
import importlib.util
import io
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


control = load("control")
worker = load("worker")
ENV = {
    "GITHUB_REPOSITORY": "jasonjiang8866/spleeterpt",
    "GITHUB_REPOSITORY_ID": "123",
    "GITHUB_RUN_ID": "456",
    "GITHUB_RUN_ATTEMPT": "1",
    "GITHUB_SHA": "a" * 40,
    "GITHUB_REF": "refs/heads/main",
}


def pod(run="456", expiry=200):
    return {
        "id": "pod" + run,
        "name": f"spt-123-{run}-1-{expiry}",
        "env": {"SPT_MANAGED_BY": ENV["GITHUB_REPOSITORY"]},
        "mounts": {},
    }


def result():
    return {
        "status": "passed",
        "report": {
            "commit": "a" * 40,
            "steps": [2, 3],
            "precision": "fp32",
            "stems": 2,
            "train_steps": [1, 2, 3],
            "validation_count": 1,
            "validation_loss": 1.0,
            "peak_allocated_bytes": 1024,
            "seconds": 1.0,
            "optimizer_present": True,
            "weights_changed": True,
            "inference_finite": True,
            "last_sha256": "b" * 64,
            "best_sha256": "c" * 64,
        },
    }


class FakeAPI:
    def __init__(self):
        self.items = []
        self.deleted = []
        self.creates = 0
        self.lose_response = False
        self.rate = "0.69"
        self.payload = None

    def pods(self):
        return self.items

    def call(self, path, method="GET", payload=None):
        if method == "POST":
            self.creates += 1
            self.payload = payload
            created = {"id": "abc", **copy.deepcopy(payload)}
            self.items.append(created)
            if self.lose_response:
                raise RuntimeError("Timed out")
            return created
        if method == "DELETE":
            self.deleted.append(path)
            return None
        return {"cost": self.rate}


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, ENV)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_v2_inventory_follows_cursor_without_losing_pods(self):
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "fake"}):
            api = control.Runpod()
        pages = [
            {"pods": [pod()], "pagination": {"hasNextPage": True, "nextCursor": "a+b/="}},
            {"pods": [pod("999")], "pagination": {"hasNextPage": False, "nextCursor": None}},
        ]
        with patch.object(api, "call", side_effect=pages) as call:
            self.assertEqual(api.pods(), [pod(), pod("999")])
            self.assertEqual(call.call_args_list[1].args, ("/pods?cursor=a%2Bb%2F%3D",))

    def test_v2_inventory_rejects_missing_or_repeated_cursor(self):
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "fake"}):
            api = control.Runpod()
        for page in ({}, {"hasNextPage": True, "nextCursor": "repeat"}):
            with (
                patch.object(api, "call", return_value={"pods": [], "pagination": page}),
                self.assertRaisesRegex(RuntimeError, "pagination"),
            ):
                api.pods()

    def test_cleanup_current_run_not_other_runs_or_resources(self):
        api = FakeAPI()
        api.items = [
            pod(),
            pod("999"),
            {**pod("777"), "env": {}},
            {**pod("888"), "name": "personal-training"},
        ]
        control.cleanup(api)
        self.assertEqual(api.deleted, ["/pods/pod456"])

    def test_reap_only_expired_owned_pods(self):
        api = FakeAPI()
        api.items = [pod(expiry=50), pod("999", expiry=200)]
        with patch.object(control.time, "time", return_value=100):
            control.cleanup(api, expired=True)
        self.assertEqual(api.deleted, ["/pods/pod456"])

    def test_refuse_persistent_storage(self):
        for extra in (
            {"mounts": {"network": [{"volumeId": "valuable", "path": "/workspace"}]}},
            {"mounts": {"global": [{"volumeId": "valuable", "path": "/workspace"}]}},
            {"mounts": {"persistent": {"size": 20, "path": "/workspace"}}},
        ):
            api = FakeAPI()
            api.items = [{**pod(), **extra}]
            with self.assertRaises(RuntimeError):
                control.cleanup(api)
            self.assertEqual(api.deleted, [])

    def test_create_lost_response_is_not_retried_and_is_cleaned(self):
        api = FakeAPI()
        api.lose_response = True
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(control, "Path", side_effect=lambda p: Path(folder) / p),
            patch.object(control, "resolve_image", return_value="image@sha256:abc"),
            self.assertRaisesRegex(RuntimeError, "Timed out"),
        ):
            control.launch(api)
        self.assertEqual(api.creates, 1)
        self.assertEqual(api.deleted, ["/pods/abc"])
        self.assertNotIn("RUNPOD_API_KEY", api.payload["env"])

    def test_price_missing_or_over_limit_releases(self):
        for rate in ("nan", "1.01", "-1"):
            api = FakeAPI()
            api.rate = rate
            with (
                tempfile.TemporaryDirectory() as folder,
                patch.object(control, "Path", side_effect=lambda p: Path(folder) / p),
                patch.object(control, "resolve_image", return_value="image"),
                self.assertRaisesRegex(RuntimeError, "rate"),
            ):
                control.launch(api)
            self.assertEqual(api.deleted, ["/pods/abc"])

    def test_success_saves_validated_results_and_releases(self):
        api = FakeAPI()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(control, "Path", side_effect=lambda p: Path(folder) / p),
            patch.object(control, "resolve_image", return_value="image"),
            patch.object(control, "request", return_value=result()),
        ):
            control.launch(api)
            self.assertEqual(
                json.loads((Path(folder) / "runpod-results/result.json").read_text()), result()
            )
        self.assertEqual(api.deleted, ["/pods/abc"])

    def test_report_pass_flag_alone_is_insufficient(self):
        control.validate_result(result(), "a" * 40)
        broken = [{"status": "passed"}]
        for key, value in [
            ("commit", "wrong"),
            ("steps", [2, 2]),
            ("validation_count", 0),
            ("weights_changed", False),
            ("last_sha256", ""),
            ("validation_loss", float("nan")),
            ("train_steps", [1, 3]),
        ]:
            record = result()
            record["report"][key] = value
            broken.append(record)
        for record in broken:
            with self.assertRaises(RuntimeError):
                control.validate_result(record, "a" * 40)

    def test_unsafe_branch_and_existing_pod_block_provision(self):
        api = FakeAPI()
        with (
            patch.dict(os.environ, {"GITHUB_REF": "refs/heads/untrusted"}),
            self.assertRaises(RuntimeError),
        ):
            control.launch(api)
        api.items = [pod()]
        with self.assertRaises(RuntimeError):
            control.launch(api)
        self.assertEqual(api.creates, 0)

    def test_worker_authentication_and_paths(self):
        server = worker.ThreadingHTTPServer(("127.0.0.1", 0), worker.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            with patch.object(worker, "TOKEN", "secret"):
                for path, token, status in [
                    ("/result", "bad", 401),
                    ("/etc/passwd", "secret", 404),
                ]:
                    req = urllib.request.Request(
                        url + path, headers={"Authorization": "Bearer " + token}
                    )
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        urllib.request.urlopen(req)
                    self.assertEqual(error.exception.code, status)
                self.assertEqual(
                    control.request(url + "/result", token="secret")["status"], "running"
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_redirect_does_not_forward_bearer(self):
        self.assertIsNone(
            control.NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.test")
        )

    def test_http_error_body_is_not_exposed(self):
        error = urllib.error.HTTPError(
            "https://service", 403, "secret body", {}, io.BytesIO(b"API_KEY")
        )
        with (
            patch.object(urllib.request.OpenerDirector, "open", side_effect=error),
            self.assertRaisesRegex(RuntimeError, "^HTTP 403 from remote service$"),
        ):
            control.request("https://service", token="API_KEY")

    def test_missing_secret_fails_before_request(self):
        with (
            patch.dict(os.environ, {"RUNPOD_API_KEY": ""}),
            patch.object(control, "request") as call,
        ):
            with self.assertRaisesRegex(RuntimeError, "missing"):
                control.Runpod()
            call.assert_not_called()


if __name__ == "__main__":
    # Mask directives from fake launches are not useful in local test output.
    with redirect_stdout(io.StringIO()):
        unittest.main()
