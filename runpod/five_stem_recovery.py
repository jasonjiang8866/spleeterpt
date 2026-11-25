"""Verify downloaded evidence in a fresh process after compute deletion; no API."""

import argparse
import hashlib
import json
import os
import tarfile
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
    cli_profile = meta.get("cuda_cli_profile", "default")
    if cli_profile not in ("default", "fft64") or (
        cli_profile == "fft64" and meta.get("cli_acceptance") is not True
    ):
        raise ValueError("Unexpected CUDA CLI precision profile in provisioning")
    if "cuda_cli" in report or "cuda-cli.json" in report["bundle"]["files"]:
        with tarfile.open(root / "listening-and-metrics.tar", "r:") as archive:
            cli = json.load(archive.extractfile("cuda-cli.json"))
        if cli.get("cli_profile", "default") != cli_profile or cli != report.get("cuda_cli"):
            raise ValueError("Archived CUDA CLI evidence differs from selected profile or report")
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
            cli_acceptance=meta.get("cli_acceptance", False),
            expected_cli_profile=cli_profile,
        )
    except ValueError:
        if not allow_rejected:
            raise
        accepted = False
    recovered = {
        "cuda_cli_profile": cli_profile,
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
    if cli_profile == "fft64":
        from precision_result import compare_cli_to_screen

        recovered["cuda_cli_screen_replay"] = (
            compare_cli_to_screen(report["cuda_cli"], meta["commit"])
            if "cuda_cli" in report
            else {"status": "missing_cli_report"}
        )
    precision = meta.get("cuda_precision_study", False)
    if precision or meta.get("cuda_settings_study"):
        filename = "cuda-precision.json" if precision else "cuda-settings.json"
        status = "cuda_precision_status" if precision else "cuda_settings_status"
        recovered[status] = "missing"
        if filename in report["bundle"]["files"]:
            with tarfile.open(root / "listening-and-metrics.tar", "r:") as archive:
                study = json.load(archive.extractfile(filename))
            if study["source_commit"] != meta["commit"] or study["release_eligible"] is not False:
                raise ValueError("CUDA settings study identity mismatch")
            if study.get("study_kind", "backend") != ("precision" if precision else "backend"):
                raise ValueError("Unexpected CUDA study kind")
            recovered[status] = "partial"
            if study["complete"]:
                cli = report["cuda_cli"]
                original = cli["calibration"]["music_original"]
                case = study["cases"]["music_original"]
                if (
                    study["checkpoint_sha256"] != cli["checkpoint_sha256"]
                    or case["input_pcm_sha256"] != original["input_pcm_sha256"]
                    or case["reference_sha256"] != original["reference_sha256"]
                ):
                    raise ValueError("CUDA settings study calibration mismatch")
                if precision:
                    from precision_result import audit

                    recovered["cuda_precision_audit"] = audit(study)
                recovered[status] = "complete"
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
