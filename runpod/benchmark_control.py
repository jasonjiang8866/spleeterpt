"""One bounded MIG launch, account key stays on Actions, compute-only cleanup."""

import json
import math
import os
import re
import secrets
import signal
import sys
import time
from pathlib import Path

from benchmark_budget import LEASE_SECONDS, RATE_CAP, check
from control import Runpod, cleanup, identity, owned, request, resolve_image
from global_storage import NAME, ensure_global, mig_regions
from gpu_profile import MIG24
from quality import ROOT

sys.path.insert(0, str(ROOT / "scripts"))


class BenchmarkDeadline(Exception):
    """Must not be swallowed by retryable transport-error handling."""


def launch(api):
    repo, repo_id = identity()
    sha = os.environ["GITHUB_SHA"]
    run, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    if (
        os.environ["GITHUB_REF"] != "refs/heads/main"
        or not re.fullmatch(r"[a-f0-9]{40}", sha)
        or not run.isdigit()
        or not attempt.isdigit()
    ):
        raise RuntimeError("Benchmark requires exact main commit and Actions identity")
    if os.environ.get("SPT_CONFIRM_BENCHMARK") != "true":
        raise RuntimeError("Explicit paid benchmark dispatch required")
    if any(owned(p, repo, repo_id) for p in api.pods()):
        raise RuntimeError("A managed Pod already exists; refusing another paid launch")
    global_volume = ensure_global(api, create=False)
    regions = mig_regions(api, MIG24)
    budget = check(api)
    image = resolve_image()
    output = Path("runpod-results")
    output.mkdir(exist_ok=True)
    token = secrets.token_urlsafe(32)
    print("::add-mask::" + token, flush=True)
    started = time.time()
    expiry = int(started) + LEASE_SECONDS
    experiment = run + "-" + attempt
    name = f"spt-{repo_id}-{run}-{attempt}-{expiry}"
    meta = {
        "commit": sha,
        "image": image,
        "pod_name": name,
        "expires_at": expiry,
        "global_volume_id": global_volume["id"],
        "budget": budget,
        "task": "benchmark",
    }

    def save():
        (output / "provision.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")

    save()
    payload = {
        "name": name,
        "image": image,
        "cloud": "SECURE",
        "gpu": {
            "id": MIG24,
            "count": 1,
            "minRamPerGpu": 16,
            "minVcpuCountPerGpu": 4,
            "minCudaVersion": "12.8",
        },
        "disk": 40,
        "mounts": {"global": [{"volumeId": global_volume["id"], "path": "/global"}]},
        "dataCenterIds": regions,
        "ports": ["8888/http"],
        "entrypoint": ["/bin/bash", "-lc"],
        "cmd": [
            "set -eu; mkdir -p /workspace; cd /workspace; git clone https://github.com/"
            + repo
            + ".git source; cd source; git checkout --detach "
            + sha
            + "; exec python -u runpod/worker.py"
        ],
        "env": {
            "SPT_MANAGED_BY": repo,
            "SPT_RESULT_TOKEN": token,
            "SPT_COMMIT": sha,
            "SPT_EXPIRY": str(expiry),
            "SPT_TASK": "benchmark",
            "SPT_GPU_ID": MIG24,
            "PYTHONUNBUFFERED": "1",
            "SPT_GLOBAL_VOLUME_ID": global_volume["id"],
            "SPT_GLOBAL_VOLUME_NAME": NAME,
            "SPT_VOLUME_ID": global_volume["id"],
            "SPT_EXPERIMENT": experiment,
        },
    }

    def deadline(*_):
        raise BenchmarkDeadline("Benchmark deadline/cancellation; releasing compute")

    previous = {
        s: signal.signal(s, deadline) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)
    }
    signal.alarm(max(1, expiry - int(time.time()) - 60))
    rate = None
    try:
        # Exactly one POST. A lost response still enters name-based cleanup.
        pod = api.call("/pods", "POST", payload)
        pod_id = pod["id"]
        if not re.fullmatch(r"[a-zA-Z0-9]+", pod_id):
            raise RuntimeError("Invalid Pod identity")
        meta["pod_id"] = pod_id
        save()
        detail = api.call("/pods/" + pod_id)
        value = detail.get("cost")
        rate = float(value) if not isinstance(value, bool) else math.nan
        if (
            not math.isfinite(rate)
            or not 0 < rate <= RATE_CAP
            or detail.get("gpu", {}).get("id") != MIG24
        ):
            rate = None
            raise RuntimeError("Actual GPU identity/rate not approved; releasing immediately")
        meta["cost_per_hour"] = rate
        save()
        print(f"Created {pod_id}: ${rate}/hour, one MIG, 60-minute lease", flush=True)
        url = f"https://{pod_id}-8888.proxy.runpod.net/result"
        last = None
        last_phase = None
        while time.time() < expiry - 120:
            try:
                result = request(url, token=token)
            except RuntimeError:
                time.sleep(15)
                continue
            if result.get("status") in {"passed", "failed"}:
                (output / "result.json").write_text(
                    json.dumps(result, indent=2, allow_nan=False) + "\n"
                )
                from benchmark_metrics import timing_summary
                from benchmark_result import validate
                from finetune_result import download_export

                validate(result, sha, global_volume["id"], pod_id, experiment)
                download_export(
                    url.removesuffix("result") + "checkpoint",
                    token,
                    result["report"],
                    output / "listening-and-metrics.tar",
                )
                meta["timing_summary"] = timing_summary(result["report"]["measurements"], rate)
                save()
                print("PASS: independent benchmark validation and export checks", flush=True)
                return
            from worker import PROGRESS

            phase = result.get("phase")
            if (
                isinstance(phase, str)
                and phase != last_phase
                and re.fullmatch(r"benchmark: command \d+/\d+", phase)
            ):
                print("Worker: " + phase, flush=True)
                last_phase = phase
            progress = result.get("progress")
            if isinstance(progress, str) and progress != last and PROGRESS.fullmatch(progress):
                print(progress, flush=True)
                last = progress
            time.sleep(15)
        raise RuntimeError("Benchmark polling deadline reached")
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        try:
            cleanup(api)
            # A successful DELETE is insufficient evidence of disappearance.
            remaining = [p for p in api.pods() if p.get("name") == name]
            if remaining:
                raise RuntimeError("Compute deletion not yet confirmed; inspect RunPod")
            meta["cleanup_confirmed"] = True
        finally:
            meta["controller_elapsed_seconds"] = time.time() - started
            meta["compute_cost_upper_estimate_usd"] = (
                rate * meta["controller_elapsed_seconds"] / 3600 if rate else None
            )
            meta["cost_scope"] = (
                "wall-clock at quoted rate; excludes persistent storage/request charges; not invoice"
            )
            save()


if __name__ == "__main__":
    try:
        launch(Runpod())
    except Exception as error:  # noqa: BLE001 -- provider text may contain private account fields
        print(
            f"ERROR: {type(error).__name__}; benchmark incomplete; inspect artifacts and cleanup",
            flush=True,
        )
        raise SystemExit(1) from None
