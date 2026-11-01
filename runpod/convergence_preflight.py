"""Read-only training inventory and pinned data/weight plan; no GPU creation."""

import json
from pathlib import Path

from benchmark_budget import check
from benchmark_preflight import observe
from control import Runpod, identity
from global_inputs import evidence


def main():
    identity()
    api = Runpod()
    resources = observe(api)
    resources["pending"] = [
        "Mounted training snapshot SHA-256 and all 500 audio hashes",
        "Official Spleeter-4 recreation hash and frozen-BN training acceptance",
        "Measured runtime projection within the single 60-minute lease",
    ]
    report = {
        "profile": "convergence",
        "resources": resources,
        "budget": check(api),
        "data": evidence()["resume"]["dataset"],
        "initial_sha256": evidence()["resume"]["initial_sha256"],
        "snapshot_sha256": "172c50e5310b22ceab3d4813a4ba1146c682db682c8aebc4c03ad235de376964",
        "snapshot_path": "/global/musdb18-training-v1/snapshot.json",
        "data_format": "MUSDB18 AAC; not HQ",
        "test_pool_used": False,
    }
    output = Path("runpod-results")
    output.mkdir(exist_ok=True)
    (output / "convergence-preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 -- do not expose provider/private account fields
        print(f"ERROR: {type(error).__name__}; read-only preflight incomplete")
        raise SystemExit(1) from None
