"""Compare independent process reports without treating one passing run as a fix."""

import argparse
import itertools
import json
from pathlib import Path


def summarize(reports):
    if len(reports) < 3 or len({r["replicate"] for r in reports}) != len(reports):
        raise ValueError("need at least three distinct replicate IDs")
    first = reports[0]
    expected_cells = {
        f"magnitude_{a}__postlogit_{b}__carrier_{c}__inverse_{d}"
        for a, b, c, d in itertools.product(("tf", "pt"), repeat=4)
    }
    expected_controls = {
        "official_masked_refeed_exact",
        "native_hybrid_replay_exact",
        "tf_post_refeed_tf_tf_exact",
        "tf_post_refeed_tf_pt_exact",
        "tf_post_refeed_pt_tf_exact",
        "tf_post_refeed_pt_pt_exact",
    }
    for report in reports:
        for field in (
            "schema_version",
            "source_commit",
            "code_sha256",
            "upstream_commit",
            "checkpoint_sha256",
            "tf_checkpoint_sha256",
            "audio_sha256",
            "config",
            "thresholds",
            "fixed_network",
            "primary_factors",
            "postlogit_scope",
            "bridge_factors",
        ):
            if report[field] != first[field]:
                raise ValueError(f"replicate provenance mismatch: {field}")
        if report["holdouts_used"] or report["release_eligible"]:
            raise ValueError("diagnostic scope violation")
        if set(report["cases"]) != {"music_original", "music_multiwindow"}:
            raise ValueError("incomplete calibration cases")
        for case, result in report["cases"].items():
            for field in (
                "decoded_pcm_sha256",
                "samples",
                "frames",
                "windows",
                "last_window_frames",
            ):
                if result[field] != first["cases"][case][field]:
                    raise ValueError(f"replicate input geometry mismatch: {case}/{field}")
            if not result["attribution_valid"] or any(
                result["controls"].get(k) is not True for k in expected_controls
            ):
                raise ValueError("replay controls invalid; do not attribute")
            if set(result["cells"]) != expected_cells:
                raise ValueError("incomplete cross cells")
            for values in result["cells"].values():
                if set(values) != set(report["config"]["instruments"]):
                    raise ValueError("incomplete stems")
    rows = []
    for case, results in first["cases"].items():
        for cell, stems in results["cells"].items():
            for stem in stems:
                values = [r["cases"][case]["cells"][cell][stem] for r in reports]
                rows.append(
                    {
                        "case": case,
                        "cell": cell,
                        "stem": stem,
                        "strict_failed_samples": [v["strict_failed_samples"] for v in values],
                        "max_absolute_error": [v["max_absolute_error"] for v in values],
                        "max_tolerance_ratio": [v["max_tolerance_ratio"] for v in values],
                        "all_replicates_passed": all(
                            v["allclose_atol_1e-5_rtol_1e-4"] for v in values
                        ),
                        "bitwise_repeatable": len({v["output_sha256"] for v in values}) == 1,
                    }
                )
    return {
        "replicates": [r["replicate"] for r in reports],
        "identical_recorded_environment": all(
            r["environment"] == first["environment"] for r in reports
        ),
        "rows": rows,
        "release_eligible": False,
        "note": "Observed repeatability on calibration only; not proof of a deterministic runtime fix. Compare differing environments separately.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize([json.loads(p.read_text()) for p in args.reports])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
