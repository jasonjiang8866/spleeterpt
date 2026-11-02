"""Verify downloaded evidence in a fresh process after compute deletion; no API."""

import argparse
import json
from pathlib import Path

from bounded_result import validate
from finetune_result import validate_export


def verify(root):
    meta = json.loads((root / "provision.json").read_text())
    if meta.get("cleanup_confirmed") is not True or meta.get("benchmark_profile") != "five-stem":
        raise ValueError("Compute deletion is not confirmed for five-stem run")
    result = json.loads((root / "result.json").read_text())
    report = result["report"]
    validate(
        result,
        meta["commit"],
        meta["global_volume_id"],
        meta["pod_id"],
        report["experiment"],
        five_stem=True,
    )
    validate_export(root / "listening-and-metrics.tar", report)
    recovered = {
        "source_commit": meta["commit"],
        "compute_deleted": True,
        "bundle_sha256": report["bundle"]["sha256"],
        "recovery": "fresh process on Actions runner after verified Pod disappearance",
        "all_members_verified": True,
        "gpu_acceptance_revalidated": True,
    }
    (root / "recovery.json").write_text(json.dumps(recovered, indent=2) + "\n")
    print("PASS: downloaded five-stem evidence recovered after compute deletion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("runpod-results"))
    verify(parser.parse_args().directory)
