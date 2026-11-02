"""Read-only, allowlisted status for this repository's active five-stem worker."""

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from benchmark_preflight import ReadOnly
from control import Runpod, identity, owned, request
from gpu_profile import ADA, MIG24
from worker import PROGRESS


def observe(api, repo, repo_id):
    api = ReadOnly(api)
    rows = []
    for pod in api.pods():
        if not owned(pod, repo, repo_id):
            continue
        env = pod.get("env", {})
        if env.get("SPT_BENCHMARK_PROFILE") != "five-stem":
            continue
        pod_id = pod.get("id", "")
        sha = env.get("SPT_COMMIT", "")
        if not re.fullmatch(r"[a-zA-Z0-9]+", pod_id) or not re.fullmatch(r"[a-f0-9]{40}", sha):
            raise ValueError("Invalid experiment identity")
        rate = pod.get("cost")
        row = {
            "pod_id": pod_id,
            "source_commit": sha,
            "gpu": env.get("SPT_GPU_ID") if env.get("SPT_GPU_ID") in {MIG24, ADA} else "unknown",
            "rate_usd_per_hour": rate
            if not isinstance(rate, bool) and isinstance(rate, (int, float)) and math.isfinite(rate)
            else None,
            "worker_status": "unreachable",
        }
        try:
            result = request(
                f"https://{pod_id}-8888.proxy.runpod.net/result",
                token=env["SPT_RESULT_TOKEN"],
            )
        except (KeyError, RuntimeError):
            rows.append(row)
            continue
        if result.get("status") in {"running", "passed", "failed"}:
            row["worker_status"] = result["status"]
        phase, progress = result.get("phase"), result.get("progress")
        if isinstance(phase, str) and re.fullmatch(r"benchmark: command \d+/\d+", phase):
            row["phase"] = phase
        if isinstance(progress, str) and PROGRESS.fullmatch(progress):
            row["progress"] = progress
        rows.append(row)
    return {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "pods": rows,
    }


if __name__ == "__main__":
    try:
        report = observe(Runpod(), *identity())
        destination = Path("runpod-results/five-stem-status.json")
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    except Exception as error:  # noqa: BLE001 -- never echo provider payloads
        print(f"ERROR: {type(error).__name__}; read-only status incomplete")
        raise SystemExit(1) from None
