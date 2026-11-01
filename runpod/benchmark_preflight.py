"""GET-only resource observation; no launch, account changes or credential export."""

import argparse
import hashlib
import json
import math
import os
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from control import Runpod, identity
from global_storage import ensure_global
from gpu_profile import MIG24
from storage import NAME as REGIONAL_NAME
from storage import valid_volume, volumes

ROOT = Path(__file__).resolve().parents[1]


class ReadOnly:
    """Reject writes even if a reused helper accidentally attempts one."""

    def __init__(self, api):
        self.api = api

    def call(self, path, method="GET", payload=None):
        if method != "GET" or payload is not None:
            raise RuntimeError("Benchmark preflight forbids API mutations")
        return self.api.call(path)

    def pods(self):
        # Reuse the paginated inventory, with every call passing the GET-only guard.
        return Runpod.pods(self)


def observe(api, *, vocal=False):
    api = ReadOnly(api)
    pods = api.pods()
    matches = [v for v in volumes(api) if v.get("name") == REGIONAL_NAME]
    if len(matches) > 1:
        raise RuntimeError("Ambiguous source volume")
    volume = valid_volume(matches[0]) if matches else None
    if volume and volume["id"] != "j3aqjpw393":
        raise RuntimeError("Unexpected source volume")
    global_volume = ensure_global(api, create=False)
    gpu = api.call(
        "/catalog/gpus/"
        + urllib.parse.quote(MIG24, safe="")
        + "?include=AVAILABILITY&product=POD&cloud=SECURE&count=1&minCudaVersion=12.8"
    )
    rate = gpu.get("price", {}).get("secure")
    valid_rate = (
        not isinstance(rate, bool)
        and isinstance(rate, (int, float))
        and math.isfinite(rate)
        and rate > 0
    )
    regions = [
        {"id": d["id"], "availability": d.get("availability")}
        for d in gpu.get("dataCenters", [])
        if isinstance(d.get("id"), str) and d.get("availability") in {"HIGH", "MEDIUM", "LOW"}
    ]
    project_pods = [
        p for p in pods if (p.get("env") or {}).get("SPT_MANAGED_BY") == "jasonjiang8866/spleeterpt"
    ]
    # Only allowlisted fields leave the runner. Never serialize env, endpoints or keys.
    return {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.environ.get("GITHUB_SHA"),
        "read_only": True,
        "account_pod_count": len(pods),
        "project_pods": [
            {k: p.get(k) for k in ("id", "name", "desiredStatus", "status", "costPerHr")}
            for p in project_pods
        ],
        "regional_volume": (
            {k: volume.get(k) for k in ("id", "name", "size", "type", "dataCenter")}
            if volume
            else None
        ),
        "regional_volume_found": volume is not None,
        "global_volume": {k: global_volume.get(k) for k in ("id", "name")},
        "gpu": {
            "id": MIG24,
            "catalog_usd_per_hour": rate if valid_rate else None,
            "within_proposed_rate_cap": valid_rate and rate <= 0.69,
            "available_regions": sorted(regions, key=lambda d: d["id"]),
            "availability_is_advisory": True,
        },
        "ready_for_paid_run": False,
        "pending": [
            "Mounted snapshot completion marker and all selected audio SHA-256 checks",
            "Official converted Spleeter weight recreation/cache verification",
            (
                "Kim Mel-Band RoFormer full checkpoint/source hashes and CPU/runtime compatibility verification"
                if vocal
                else "HTDemucs full checkpoint hash and CPU/runtime compatibility verification"
            ),
            "Account credit and current-month spend (not exposed by this inventory)",
            "Benchmark worker, hard deadline, cleanup and independent result validation",
        ],
    }


def data_plan(*, vocal=False):
    provenance_path = ROOT / "docs/results/musdb18-training-provenance.json"
    p = json.loads(provenance_path.read_text())
    prior = json.loads((ROOT / "docs/results/persistent-pilot-41189465452.json").read_text())
    tracks = sorted(p["split_track_ids"]["validation"])
    if len(tracks) != 10 or len(set(tracks)) != 10 or p["test_audio_extracted"] is not False:
        raise ValueError("Expected pinned ten-song screen with no extracted test audio")
    selected = [t for t in p["tracks"] if t["id"].removesuffix(".stem.mp4") in tracks]
    if len(selected) != 10:
        raise ValueError("Incomplete validation inventory")
    return {
        "dataset": "MUSDB18 AAC decoded to WAV; not MUSDB18-HQ",
        "provenance_record_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        "original_provenance_sha256": prior["resume"]["dataset"]["provenance_sha256"],
        "archive_sha256": p["archive_sha256"],
        "track_ids": tracks,
        "samples": sum(t["samples"] for t in selected),
        "sample_rate": 44100,
        "expected_audio_files": 50,
        "test_pool_used": False,
        "upstream_training_overlap": "unresolved; project validation is not a shared held-out set",
        "spleeter_initial_sha256": (
            json.loads((ROOT / "docs/results/spleeter2-cpu-preflight.json").read_text())[
                "converted_sha256"
            ]
            if vocal
            else prior["resume"]["initial_sha256"]
        ),
        "evidence_scope": "repository provenance only; not a current mounted-file check",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("four-stem", "vocal"), default="four-stem")
    args = parser.parse_args()
    identity()
    vocal = args.profile == "vocal"
    report = {
        "profile": args.profile,
        "resources": observe(Runpod(), vocal=vocal),
        "data": data_plan(vocal=vocal),
    }
    out = ROOT / "runpod-results"
    out.mkdir(exist_ok=True)
    (out / "benchmark-preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 -- never echo provider data/credentials
        print(f"ERROR: {type(error).__name__}; read-only preflight incomplete")
        raise SystemExit(1) from None
