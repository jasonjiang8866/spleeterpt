"""Verify downloaded evidence in a fresh process after compute deletion; no API."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from bounded_result import validate
from finetune_result import validate_export
from gpu_profile import validate_runtime


def verify(root, *, allow_rejected=False):
    meta = json.loads((root / "provision.json").read_text())
    if meta.get("cleanup_confirmed") is not True or meta.get("benchmark_profile") != "five-stem":
        raise ValueError("Compute deletion is not confirmed for five-stem run")
    result = json.loads((root / "result.json").read_text())
    report = result["report"]
    if (
        report.get("commit") != meta["commit"]
        or report.get("pod_id") != meta["pod_id"]
        or report.get("global_volume_id") != meta["global_volume_id"]
        or report.get("kind") != "five-stem-bounded-v1"
    ):
        raise ValueError("Recovered experiment identity mismatch")
    validate_runtime(report["gpu_runtime"], meta["gpu_id"])
    validate_export(root / "listening-and-metrics.tar", report)
    accepted = True
    try:
        validate(
            result,
            meta["commit"],
            meta["global_volume_id"],
            meta["pod_id"],
            report["experiment"],
            five_stem=True,
            expected_gpu=meta["gpu_id"],
            expected_fft_policy=meta.get("fft_batch_policy", "native"),
        )
    except ValueError:
        if not allow_rejected:
            raise
        accepted = False
    recovered = {
        "source_commit": meta["commit"],
        "fft_batch_policy": meta.get("fft_batch_policy", "native"),
        "release_eligible": False,
        "compute_deleted": True,
        "bundle_sha256": report["bundle"]["sha256"],
        "recovery": "fresh process after verified Pod disappearance",
        "execution_location": "GitHub Actions"
        if os.environ.get("GITHUB_ACTIONS") == "true"
        else "local",
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "all_members_verified": True,
        "gpu_acceptance_revalidated": accepted,
        "acceptance_status": "accepted" if accepted else "rejected",
    }
    (root / "recovery.json").write_text(json.dumps(recovered, indent=2) + "\n")
    print(f"PASS: archive recovery after compute deletion; GPU acceptance={accepted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("runpod-results"))
    parser.add_argument(
        "--allow-rejected",
        action="store_true",
        help="Recover rejected evidence without marking GPU acceptance passed",
    )
    args = parser.parse_args()
    verify(args.directory, allow_rejected=args.allow_rejected)
