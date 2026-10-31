"""Runpod orchestration. The account credential stays on the Actions runner."""

import argparse
import json
import math
import os
import re
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.runpod.io/v2"
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
GPU = "NVIDIA GeForce RTX 4090"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(url, *, token=None, method="GET", payload=None, limit=8_000_000):
    headers = {"Content-Type": "application/json", "User-Agent": "spleeterpt-runpod-actions/0.1"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(
        url,
        headers=headers,
        method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
    )
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=30) as response:
            raw = response.read(limit + 1)
            if len(raw) > limit:
                raise RuntimeError("Response exceeds size limit")
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        # Never print response bodies: provider errors can echo env credentials.
        raise RuntimeError(f"HTTP {error.code} from remote service") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("Remote service unavailable or timed out") from None


class Runpod:
    def __init__(self):
        self.key = os.environ.get("RUNPOD_API_KEY", "")
        if not self.key:
            raise RuntimeError("RUNPOD_API_KEY Secret is missing")

    def call(self, path, method="GET", payload=None):
        return request(API + path, token=self.key, method=method, payload=payload)

    def pods(self):
        items, seen = [], set()
        path = "/pods"
        while True:
            result = self.call(path)
            if not isinstance(result, dict) or not isinstance(result.get("pods"), list):
                raise TypeError("Unexpected Pod list response")
            items.extend(result["pods"])
            page = result.get("pagination", {})
            if page.get("hasNextPage") is False:
                return items
            cursor = page.get("nextCursor")
            if not isinstance(cursor, str) or not cursor or cursor in seen:
                raise RuntimeError("Invalid Pod pagination; refusing incomplete inventory")
            seen.add(cursor)
            path = "/pods?" + urllib.parse.urlencode({"cursor": cursor})


def identity():
    repo = os.environ["GITHUB_REPOSITORY"]
    repo_id = os.environ["GITHUB_REPOSITORY_ID"]
    if repo != "jasonjiang8866/spleeterpt" or not repo_id.isdigit():
        raise RuntimeError("This workflow is restricted to jasonjiang8866/spleeterpt")
    return repo, repo_id


def owned(pod, repo, repo_id):
    match = re.fullmatch(rf"spt-{re.escape(repo_id)}-(\d+)-(\d+)-(\d+)", pod.get("name", ""))
    if not match or pod.get("env", {}).get("SPT_MANAGED_BY") != repo:
        return None
    if any((pod.get("mounts") or {}).values()):
        raise RuntimeError("Refusing to delete managed Pod with persistent storage")
    return match


def cleanup(api, *, expired=False):
    repo, repo_id = identity()
    failures = []
    removed = 0
    for pod in api.pods():
        match = owned(pod, repo, repo_id)
        if not match:
            continue
        run, attempt, expiry = match.groups()
        selected = (
            int(expiry) <= time.time()
            if expired
            else (
                run == os.environ["GITHUB_RUN_ID"] and attempt == os.environ["GITHUB_RUN_ATTEMPT"]
            )
        )
        if selected:
            try:
                pod_id = pod["id"]
                if not re.fullmatch(r"[a-zA-Z0-9]+", pod_id):
                    raise RuntimeError("Invalid Pod id")
                api.call("/pods/" + pod_id, "DELETE")
                removed += 1
            except RuntimeError:
                failures.append(pod["id"])
    print(f"Cleanup: {removed} managed Pod(s) deleted", flush=True)
    if failures:
        raise RuntimeError(
            "Cleanup failed; check Runpod console for Pod(s): " + ", ".join(failures)
        )


def validate_result(result, sha):
    if not isinstance(result, dict) or result.get("status") != "passed":
        raise RuntimeError("GPU worker did not report success")
    report = result.get("report", {})
    if report.get("commit") != sha or report.get("steps") != [2, 3]:
        raise RuntimeError("Result commit or resume cursor mismatch")
    if report.get("precision") != "fp32" or report.get("stems") != 2:
        raise RuntimeError("Unexpected model configuration")
    if report.get("train_steps") != [1, 2, 3] or report.get("validation_count", 0) < 1:
        raise RuntimeError("Training/validation records are incomplete")
    for name in ("peak_allocated_bytes", "seconds", "validation_loss"):
        value = report.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("Missing numeric result: " + name)
        if not math.isfinite(value) or value < 0 or (name != "validation_loss" and value == 0):
            raise RuntimeError("Invalid numeric result: " + name)
    for name in ("optimizer_present", "weights_changed", "inference_finite"):
        if report.get(name) is not True:
            raise RuntimeError("Result check failed: " + name)
    for name in ("last_sha256", "best_sha256"):
        if not re.fullmatch(r"[a-f0-9]{64}", report.get(name, "")):
            raise RuntimeError("Missing checkpoint checksum")


def resolve_image():
    # Resolve the public versioned image to an immutable registry digest before spending.
    auth = request(
        "https://auth.docker.io/token?service=registry.docker.io&scope=repository:runpod/pytorch:pull"
    )
    req = urllib.request.Request(
        "https://registry-1.docker.io/v2/runpod/pytorch/manifests/" + IMAGE.split(":", 1)[1],
        headers={
            "Authorization": "Bearer " + auth["token"],
            "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json",
        },
        method="HEAD",
    )
    with urllib.request.build_opener(NoRedirect).open(req, timeout=30) as response:
        digest = response.headers.get("Docker-Content-Digest", "")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise RuntimeError("Cannot resolve container image digest")
    return "runpod/pytorch@" + digest


def launch(api):
    repo, repo_id = identity()
    if os.environ.get("GITHUB_REF") != "refs/heads/main":
        raise RuntimeError("GPU launches must use main")
    sha = os.environ["GITHUB_SHA"]
    if not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise RuntimeError("Invalid source commit")
    run, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    if not run.isdigit() or not attempt.isdigit():
        raise RuntimeError("Invalid Actions identity")
    # Do not retry provisioning: a timed-out POST may already have created a paid Pod.
    if any(owned(p, repo, repo_id) for p in api.pods()):
        raise RuntimeError("A managed Pod already exists; run cleanup before another smoke test")
    image = resolve_image()
    expiry = int(time.time()) + 1800
    name = f"spt-{repo_id}-{run}-{attempt}-{expiry}"
    token = secrets.token_urlsafe(32)
    print("::add-mask::" + token, flush=True)
    output = Path("runpod-results")
    output.mkdir(exist_ok=True)
    metadata = {"commit": sha, "pod_name": name, "image": image, "expires_at": expiry}
    (output / "provision.json").write_text(json.dumps(metadata, indent=2))
    bootstrap = (
        "set -eu; mkdir -p /workspace; cd /workspace; git clone https://github.com/"
        + repo
        + ".git source; "
        "cd source; git checkout --detach " + sha + "; exec python -u runpod/worker.py"
    )
    payload = {
        "name": name,
        "image": image,
        "cloud": "SECURE",
        "gpu": {
            "id": GPU,
            "count": 1,
            "minRamPerGpu": 16,
            "minVcpuCountPerGpu": 4,
            "minCudaVersion": "12.8",
        },
        "disk": 30,
        "mounts": {},
        "ports": ["8888/http"],
        "entrypoint": ["/bin/bash", "-lc"],
        "cmd": [bootstrap],
        "env": {
            "SPT_MANAGED_BY": repo,
            "SPT_RESULT_TOKEN": token,
            "SPT_COMMIT": sha,
            "SPT_EXPIRY": str(expiry),
            "PYTHONUNBUFFERED": "1",
        },
    }
    try:
        pod = api.call("/pods", "POST", payload)
        pod_id = pod["id"]
        if not re.fullmatch(r"[a-zA-Z0-9]+", pod_id):
            raise RuntimeError("Invalid Pod id")
        metadata["pod_id"] = pod_id
        (output / "provision.json").write_text(json.dumps(metadata, indent=2))
        # Actual price is returned only after creation; fail closed on missing/high rates.
        detail = api.call("/pods/" + pod_id)
        rate = float(detail.get("cost", "nan"))
        if not math.isfinite(rate) or not 0 < rate <= 1.0:
            raise RuntimeError("Pod rate missing or above $1/hour; releasing immediately")
        metadata["cost_per_hour"] = rate
        (output / "provision.json").write_text(json.dumps(metadata, indent=2))
        print(f"Created {pod_id}: ${rate}/hour; lease 30 minutes", flush=True)
        url = f"https://{pod_id}-8888.proxy.runpod.net/result"
        while time.time() < expiry - 90:
            try:
                result = request(url, token=token)
            except RuntimeError:
                time.sleep(15)
                continue
            if result.get("status") in {"passed", "failed"}:
                (output / "result.json").write_text(json.dumps(result, indent=2))
                validate_result(result, sha)
                print("PASS: full-size CUDA train/validation/resume/inference", flush=True)
                return
            time.sleep(15)
        raise RuntimeError("GPU smoke deadline exceeded; check artifacts and Runpod console")
    finally:
        # Name-based discovery also handles POST success with a lost response.
        cleanup(api)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "smoke", "cleanup", "reap"])
    args = parser.parse_args()
    identity()
    api = Runpod()
    if args.action == "preflight":
        api.pods()
        print("PASS: RUNPOD_API_KEY authenticates to REST v2 Pod management (read-only check).")
    elif args.action == "smoke":
        launch(api)
    else:
        cleanup(api, expired=args.action == "reap")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 -- sanitize all top-level credential-bearing errors
        # Do not print provider responses, environment, or credential-bearing tracebacks.
        print(
            f"ERROR: {error}"
            if isinstance(error, RuntimeError)
            else f"ERROR: {type(error).__name__}; inspect sanitized artifacts and provider console"
        )
        raise SystemExit(1) from None
