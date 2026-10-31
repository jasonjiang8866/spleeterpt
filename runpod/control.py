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
        from storage import safe_mounts

        if not safe_mounts(pod):
            raise RuntimeError(
                "Refusing to delete managed Pod with unrecognized persistent storage"
            )
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


def launch(api, task="smoke"):
    if task not in {
        "smoke",
        "quality",
        "finetune",
        "training",
        "recovery-export",
        "recovery-resume",
        "persistent-prepare",
        "persistent-train",
        "persistent-resume",
    }:
        raise RuntimeError("Unsupported GPU task")
    repo, repo_id = identity()
    if os.environ.get("GITHUB_REF") != "refs/heads/main":
        raise RuntimeError("GPU launches must use main")
    sha = os.environ["GITHUB_SHA"]
    if not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise RuntimeError("Invalid source commit")
    source_commit = os.environ.get("SPT_SOURCE_COMMIT", "")
    if source_commit and (
        task != "persistent-resume" or not re.fullmatch(r"[a-f0-9]{40}", source_commit)
    ):
        raise RuntimeError("Invalid persistent source commit")
    experiment = os.environ.get("SPT_EXPERIMENT", "")
    if experiment and not re.fullmatch(r"[0-9]+-[0-9]+", experiment):
        raise RuntimeError("Invalid persistent experiment")
    if task == "finetune" and experiment != "41189465452-1":
        raise RuntimeError("Fine-tune validation requires the pinned step-90 experiment")
    restore_source = None
    if task == "recovery-resume":
        from recovery_result import validate_bundle, validate_recovery

        restored = json.loads(Path("restore-input/result.json").read_text())
        validate_recovery(restored, sha, "export")
        restore_source = restored["report"]
        validate_bundle(
            Path("restore-input/checkpoint.tar"), restore_source["bundle"], complete=False
        )
    run, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    if not run.isdigit() or not attempt.isdigit():
        raise RuntimeError("Invalid Actions identity")
    # Do not retry provisioning: a timed-out POST may already have created a paid Pod.
    if any(owned(p, repo, repo_id) for p in api.pods()):
        raise RuntimeError("A managed Pod already exists; run cleanup before another smoke test")
    volume = None
    if task.startswith("persistent-") or task == "finetune":
        from storage import ensure_volume

        volume = ensure_volume(api, create=task == "persistent-prepare")
        if task == "finetune" and volume["id"] != "j3aqjpw393":
            raise RuntimeError("Fine-tune validation requires the verified data volume")
    image = resolve_image()
    expiry = int(time.time()) + 1800
    name = f"spt-{repo_id}-{run}-{attempt}-{expiry}"
    token = secrets.token_urlsafe(32)
    print("::add-mask::" + token, flush=True)
    output = Path("runpod-results")
    output.mkdir(exist_ok=True)
    metadata = {"commit": sha, "pod_name": name, "image": image, "expires_at": expiry, "task": task}
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
            "SPT_TASK": task,
            "PYTHONUNBUFFERED": "1",
        },
    }
    if volume:
        metadata["volume"] = volume
        payload["mounts"] = {"network": [{"volumeId": volume["id"], "path": "/data"}]}
        payload["dataCenterIds"] = [volume["dataCenter"]]
        payload["env"].update(
            SPT_VOLUME_ID=volume["id"],
            SPT_VOLUME_NAME=volume["name"],
            SPT_EXPERIMENT=os.environ.get("SPT_EXPERIMENT", run + "-" + attempt),
        )
        if source_commit:
            payload["env"]["SPT_SOURCE_COMMIT"] = source_commit
        (output / "provision.json").write_text(json.dumps(metadata, indent=2))
    if restore_source:
        payload["env"]["SPT_RESTORE_DESCRIPTOR"] = json.dumps(restore_source["bundle"])
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
        last_phase = None
        last_progress = None
        restore_uploaded = False
        transfer_probed = False
        while time.time() < expiry - 90:
            try:
                result = request(url, token=token)
            except RuntimeError:
                time.sleep(15)
                continue
            if (task.startswith("recovery-") or task == "finetune") and not transfer_probed:
                from transfer import probe_transfer

                probe_transfer(url.removesuffix("result") + "checkpoint", token)
                transfer_probed = True
                print("PASS: authenticated checkpoint transport probe", flush=True)
            if restore_source and not restore_uploaded:
                from transfer import upload_bundle

                upload_bundle(
                    url.removesuffix("result") + "restore",
                    token,
                    Path("restore-input/checkpoint.tar"),
                    restore_source["bundle"],
                )
                restore_uploaded = True
                print("Restore checkpoint delivered and SHA-256 verified", flush=True)
            progress = result.get("progress")
            if isinstance(progress, str) and progress != last_progress:
                from worker import PROGRESS

                if PROGRESS.fullmatch(progress):
                    print(progress, flush=True)
                    last_progress = progress
            phase = result.get("phase")
            if isinstance(phase, str) and phase != last_phase:
                print("Worker: " + phase[:100], flush=True)
                last_phase = phase
            if result.get("status") in {"passed", "failed"}:
                (output / "result.json").write_text(json.dumps(result, indent=2))
                if task.startswith("persistent-"):
                    from persistent_result import validate_persistent

                    validate_persistent(result, sha, task, volume["id"], pod_id)
                    if result["report"].get("experiment") != payload["env"]["SPT_EXPERIMENT"]:
                        raise RuntimeError("Persistent experiment mismatch")
                    if task == "persistent-resume" and result["report"].get("source_commit") != (
                        source_commit or sha
                    ):
                        raise RuntimeError("Persistent source commit mismatch")
                elif task == "finetune":
                    from finetune_result import download_export, validate_finetune

                    validate_finetune(result, sha, volume["id"], pod_id)
                    download_export(
                        url.removesuffix("result") + "checkpoint",
                        token,
                        result["report"],
                        output / "listening-and-metrics.tar",
                    )
                    print(
                        "PASS: ten full validation songs, paired metrics and listening export verified",
                        flush=True,
                    )
                elif task == "quality":
                    from quality_result import validate_quality

                    validate_quality(result, sha, output / "quality")
                    print("PASS: complete CUDA quality pilot (not a full benchmark)", flush=True)
                elif task == "training":
                    from training_result import validate_training

                    validate_training(result, sha)
                elif task.startswith("recovery-"):
                    from recovery_result import validate_recovery
                    from transfer import download_bundle

                    stage = task.split("-", 1)[1]
                    validate_recovery(result, sha, stage, restore_source)
                    if result["report"]["pod_id"] != pod_id:
                        raise RuntimeError("Recovery Pod identity mismatch")
                    download_bundle(
                        url.removesuffix("result") + "checkpoint",
                        token,
                        result["report"]["bundle"],
                        output / "checkpoint.tar",
                        complete=stage == "resume",
                    )
                    print("PASS: checkpoint downloaded and every member hash verified", flush=True)
                else:
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
    parser.add_argument(
        "action",
        choices=[
            "preflight",
            "smoke",
            "quality",
            "finetune",
            "training",
            "recovery-export",
            "recovery-resume",
            "persistent-prepare",
            "persistent-train",
            "persistent-resume",
            "cleanup",
            "reap",
        ],
    )
    args = parser.parse_args()
    identity()
    api = Runpod()
    if args.action == "preflight":
        api.pods()
        print("PASS: RUNPOD_API_KEY authenticates to REST v2 Pod management (read-only check).")
    elif args.action in {
        "smoke",
        "quality",
        "finetune",
        "training",
        "recovery-export",
        "recovery-resume",
        "persistent-prepare",
        "persistent-train",
        "persistent-resume",
    }:
        launch(api, args.action)
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
