"""Render an accepted training report; raw inputs remain in the verified run export."""

import argparse
import copy
import hashlib
import json
import statistics as stats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runpod"))
from convergence_evaluation import ARMS, MILESTONES, STEMS
from convergence_result import validate_report
from musdb_evaluation import track_score


def aggregate(result, provision, source_files):
    r = result["report"]
    validate_report(r, r["commit"], r["storage"]["global_volume_id"], r["pod_id"], r["experiment"])
    if (
        provision["commit"] != r["commit"]
        or provision["pod_id"] != r["pod_id"]
        or provision["benchmark_profile"] != "convergence"
        or provision["cleanup_confirmed"] is not True
    ):
        raise ValueError("Cleanup/provision identity mismatch")
    out = {
        k: copy.deepcopy(r[k])
        for k in (
            "kind",
            "commit",
            "pod_id",
            "experiment",
            "storage",
            "dataset",
            "protocol",
            "initial_sha256",
            "initial_bn_fingerprint",
            "torch",
            "cuda",
            "gpu_runtime",
            "museval",
            "coverage",
            "curves",
            "summary",
            "budget_projections",
            "matched_rng",
            "weights_unchanged",
            "complete",
            "seconds",
            "listening",
            "bundle",
        )
    }
    out["kind"] = "convergence-pilot-summary"
    out["worker_status"] = result["status"]
    out["data_audit"] = "retained report revalidated with finite nonnegative loss/gradient rule"
    out["listening_export_retrieved"] = False
    out["export_bytes_independently_verified"] = False
    out["run_url"] = (
        "https://github.com/jasonjiang8866/spleeterpt/actions/runs/" + r["experiment"].split("-")[0]
    )
    out["compute"] = {
        k: provision[k]
        for k in (
            "cost_per_hour",
            "controller_elapsed_seconds",
            "compute_cost_upper_estimate_usd",
            "cleanup_confirmed",
            "cost_scope",
            "budget",
        )
    }
    out["source_files"] = {
        name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for name, p in source_files.items()
    }
    out["arms"] = {}
    baseline = {t["id"]: t["scores"] for t in r["curves"]["official_converted"]["tracks"]}
    for arm, proofs in r["arms"].items():
        history = proofs[-1]["metrics"]
        epochs = []
        for epoch in range(10):
            training = [m for m in history if m["kind"] == "train" and m["epoch"] == epoch]
            validation = [m for m in history if m["kind"] == "validation" and m["epoch"] == epoch]
            epochs.append(
                {
                    "epoch": epoch + 1,
                    "step": (epoch + 1) * 90,
                    "train_loss_mean": stats.mean(m["loss"] for m in training),
                    "validation_loss": validation[0]["loss"],
                    "gradient_norm_median": stats.median(m["grad_norm"] for m in training),
                    "gradient_norm_max": max(m["grad_norm"] for m in training),
                }
            )
        out["arms"][arm] = {
            "learning_rate": ARMS[arm],
            "epochs": epochs,
            "phase_seconds_total": sum(p["timing"]["seconds"] for p in proofs),
            "training_peak_allocated_bytes": max(
                p["timing"]["peak_allocated_bytes"] for p in proofs
            ),
            "milestones": [
                {k: copy.deepcopy(v) for k, v in p.items() if k != "metrics"} for p in proofs
            ],
            "si_sdr_paired_median_curve": {
                s: [0.0]
                + [
                    stats.median(
                        t["scores"][s] - baseline[t["id"]][s]
                        for t in r["curves"][f"{arm}_step{step}"]["tracks"]
                    )
                    for step in MILESTONES
                ]
                for s in STEMS
            },
        }
    out["per_track"] = [
        {
            "id": t["id"],
            "samples": t["decoded_samples"],
            "metrics": {
                metric: {
                    stem: {v: track_score(t, v, stem, metric) for v in t["scores"]}
                    for stem in STEMS
                }
                for metric in ("bsseval_v4_sdr_db", "global_stereo_si_sdr_db")
            },
        }
        for t in r["records"]
    ]
    out["coverage_summary"] = {
        "train_audio_samples": sum(t["samples"] for t in r["coverage"]["tracks"]),
        "unique_covered_samples": sum(t["covered_samples"] for t in r["coverage"]["tracks"]),
        "crop_samples_with_repetition": r["coverage"]["crop_samples"] * 900,
    }
    return out


def figure(data, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.hashsalt": "spleeterpt-convergence",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    colors = {"lr1e5": "#2267ae", "lr3e6": "#d46a21"}
    for arm, values in data["arms"].items():
        label = f"LR {ARMS[arm]:g}"
        x = [e["step"] for e in values["epochs"]]
        for axis, field in zip(axes[0], ("train_loss_mean", "validation_loss"), strict=True):
            axis.plot(
                x,
                [e[field] for e in values["epochs"]],
                marker="o",
                markersize=3,
                color=colors[arm],
                label=label,
            )
    for axis, title in zip(
        axes[0], ("Training loss (epoch mean)", "Validation loss (fixed crops)"), strict=True
    ):
        axis.set(title=title, xlabel="Optimizer updates", ylabel="Summed magnitude L1")
        axis.legend(frameon=False)
    all_deltas = [
        d
        for a in data["arms"].values()
        for s in a["si_sdr_paired_median_curve"].values()
        for d in s
    ]
    lo, hi = min(all_deltas), max(all_deltas)
    pad = max((hi - lo) * 0.12, 0.02)
    for axis, stem in zip(axes[1:].flat, STEMS, strict=True):
        for arm, values in data["arms"].items():
            axis.plot(
                [0, *MILESTONES],
                values["si_sdr_paired_median_curve"][stem],
                marker="o",
                color=colors[arm],
                label=f"LR {ARMS[arm]:g}",
            )
        axis.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
        axis.set(
            title=stem.capitalize() + " — paired SI-SDR change",
            xlabel="Optimizer updates",
            ylabel="Median paired delta (dB)",
            ylim=(lo - pad, hi + pad),
        )
        axis.set_xticks([0, *MILESTONES])
    for axis in axes.flat:
        axis.grid(alpha=0.18)
    fig.suptitle("Frozen-BN Spleeter-4 fine-tuning", fontsize=19, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.953,
        "MUSDB18 AAC · 90 training / 10 validation songs · two learning rates · one seed",
        ha="center",
        color="#555555",
    )
    fig.text(
        0.06,
        0.025,
        "SI-SDR: full stereo songs; median of within-song changes versus official weights. Higher is better.\n"
        "Loss uses crops and a different objective. Convergence is unproven; 50 test songs unused.\n"
        "Metrics revalidated offline after a zero-gradient validator fix; workflow failed and listening export was not retrieved.",
        color="#555555",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.98, 0.94), h_pad=2.1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, metadata={"Date": None})
    fig.savefig(destination.with_suffix(".png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--provision", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figure", required=True, type=Path)
    args = parser.parse_args()
    data = aggregate(
        json.loads(args.result.read_text()),
        json.loads(args.provision.read_text()),
        {"result.json": args.result, "provision.json": args.provision},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    figure(data, args.figure)


if __name__ == "__main__":
    main()
