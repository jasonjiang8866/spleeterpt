"""One serial CPU3 experiment, with verified teardown between different locations."""

import json
import math
import os
import re
import secrets
import time
from pathlib import Path

from benchmark_budget import check
from control import Runpod, cleanup, identity, owned, request, resolve_image
from native_intel_worker import SOURCE

OUT = Path("runpod-results")
RATE = 0.16
LIMIT_SECONDS = 2400


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def locations(api):
    catalog = api.call("/catalog/cpus?include=AVAILABILITY&product=POD&vcpuCount=4")
    item = next(item for item in catalog["cpus"] if item["id"] == "cpu3g")
    if not 0 < item["price"]["securePerVcpu"] * 4 <= RATE:
        raise ValueError("CPU3 quote above the rate limit")
    return sorted(
        {
            dc["id"]
            for dc in item.get("dataCenters", [])
            if dc.get("availability") in {"HIGH", "MEDIUM", "LOW"}
        }
    )


def validate_report(report, model):
    pins = json.loads(Path("scripts/five_stem_source.json").read_text())
    history = json.loads(
        Path("docs/results/five-stem-20261102/5stems-strict-final.json").read_text()
    )
    cases = report["calibration"]
    if (
        report["source_commit"] != SOURCE
        or report["upstream_commit"] != pins["upstream_commit"]
        or report["checkpoint_sha256"] != pins["converted_sha256"]
        or report["tf_checkpoint_files"] != history["tf_checkpoint_files"]
        or report["environment"]["cpu_model"] != model
        or report["thresholds"] != {"atol": 1e-5, "rtol": 1e-4}
        or set(cases) != set(history["case_provenance"])
        or len(cases) != 5
        or report["calibration_passed"] is not True
    ):
        raise ValueError("Frozen calibration provenance or verdict mismatch")
    for name, case in cases.items():
        if case["input_pcm_sha256"] != history["case_provenance"][name]["decoded_pcm_sha256"]:
            raise ValueError("Calibration PCM mismatch")
        if set(case["candidate"]) != set(pins["instruments"]) or not all(
            row["allclose_atol_1e-5_rtol_1e-4"] is True for row in case["candidate"].values()
        ):
            raise ValueError("Incomplete or failing five-stem calibration")


def run():
    repo, repo_id = identity()
    if os.environ.get("CONFIRM_RENTAL") != "true" or os.environ["GITHUB_RUN_ATTEMPT"] != "1":
        raise ValueError("Requires a fresh confirmed manual dispatch")
    run_id, sha = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"]
    if not run_id.isdigit() or not re.fullmatch("[a-f0-9]{40}", sha):
        raise ValueError("Invalid run identity")
    api = Runpod()
    OUT.mkdir(exist_ok=True)
    check(api)
    if any(owned(p, repo, repo_id) for p in api.pods()):
        raise RuntimeError("A managed Pod already exists")
    image = resolve_image("library/python:3.12-bookworm")
    regions = locations(api)[:3]
    deadline = int(time.time()) + LIMIT_SECONDS
    for index, region in enumerate(regions, 1):
        if time.time() + 180 >= deadline:
            break
        name = f"spt-{repo_id}-{run_id}-1-{deadline - index}"
        token = secrets.token_urlsafe(32)
        payload = {
            "name": name,
            "image": image,
            "cloud": "SECURE",
            "cpu": {"id": "cpu3g", "vcpuCount": 4},
            "disk": 20,
            "mounts": {},
            "dataCenterIds": [region],
            "ports": ["8888/http"],
            "entrypoint": ["/bin/bash", "-ec"],
            "cmd": [
                (
                    f"git clone -q https://github.com/{repo}.git /tmp/controller && "
                    f"cd /tmp/controller && git checkout -q {sha} && exec python -u runpod/native_intel_worker.py"
                )
            ],
            "env": {"SPT_MANAGED_BY": repo, "SPT_RESULT_TOKEN": token, "SPT_TASK": "native-intel"},
        }
        audit = {"region": region, "image": image, "start_unix": int(time.time()), "name": name}
        try:
            pod = api.call("/pods", "POST", payload)
            pod_id = pod["id"]
            if not re.fullmatch("[A-Za-z0-9]+", pod_id):
                raise ValueError("Invalid Pod id")
            detail = api.call("/pods/" + pod_id)
            rate = float(detail["cost"])
            if (
                not math.isfinite(rate)
                or not 0 < rate <= RATE
                or detail.get("cpu", {}).get("id") != "cpu3g"
            ):
                raise ValueError("Unexpected assigned hardware or price")
            audit.update(id=pod_id, hourly_usd=rate)
            probe_deadline = min(deadline, time.time() + 300)
            while time.time() < probe_deadline:
                try:
                    result = request(f"https://{pod_id}-8888.proxy.runpod.net/result", token=token)
                except RuntimeError:
                    time.sleep(8)
                    continue
                cpu = result.get("cpu") or {}
                status = result["status"]
                if cpu.get("vendor") == "GenuineIntel":
                    probe_deadline = deadline
                if status in {"other-cpu", "passed", "failed"}:
                    audit.update(cpu=cpu, status=status)
                    if status == "other-cpu" and cpu.get("vendor") == "AuthenticAMD":
                        break
                    write("native-result.json", result)
                    if status != "passed" or cpu.get("vendor") != "GenuineIntel":
                        raise RuntimeError("Intel calibration failed; inspect artifact")
                    validate_report(result["report"], cpu["model"])
                    return
                time.sleep(8)
            else:
                raise TimeoutError("CPU probe or calibration timed out")
        finally:
            cleanup(api)
            for _ in range(10):
                if all(p.get("name") != name for p in api.pods()):
                    break
                time.sleep(3)
            else:
                raise RuntimeError("Pod deletion has not been verified")
            audit.update(cleanup_confirmed=True, end_unix=int(time.time()))
            write(f"attempt-{index}.json", audit)
    raise RuntimeError("No Intel calibration completed within the placement limit")


if __name__ == "__main__":
    run()
