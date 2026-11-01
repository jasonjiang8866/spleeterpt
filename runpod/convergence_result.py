"""Independent CPU acceptance of the bounded training screen (no Torch required)."""

import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

from global_inputs import encoded, evidence
from gpu_profile import MIG24, validate_runtime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import itertools

from convergence_evaluation import ARMS, MILESTONES, PROTOCOL, SNAPSHOT_SHA256, VARIANTS, summarize
from finetune_evaluation import RATE, STEMS, validation_inventory


def positive(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


def nonnegative(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def validate_training(r):
    if set(r["arms"]) != set(ARMS):
        raise ValueError("Missing training arm")
    signatures = []
    for arm, lr in ARMS.items():
        proofs = r["arms"][arm]
        if len(proofs) != len(MILESTONES):
            raise ValueError("Incomplete milestones")
        previous = None
        for proof, step in zip(proofs, MILESTONES, strict=True):
            epoch = step // 90
            if (
                proof["cursor"] != [step, epoch, 0]
                or proof["optimizer_steps"] != [step]
                or not all(
                    proof[k] is True
                    for k in (
                        "finite",
                        "bn_buffers_unchanged",
                        "bn_affine_changed",
                        "parameters_changed",
                    )
                )
                or proof["bn_fingerprint"] != r["initial_bn_fingerprint"]
                or proof["resume_sha256"] != previous
            ):
                raise ValueError("Training state/resume acceptance failed")
            if step != 900 and proof.get("global_restore_equal") is not True:
                raise ValueError("Missing verified Global round-trip")
            recipe = {
                k: v
                for k, v in PROTOCOL["recipe"].items()
                if k not in ("epochs", "workers", "optimizer", "save_every")
            }
            signature = proof["signature"]
            if (
                any(signature[k] != v for k, v in recipe.items())
                or signature["lr"] != lr
                or any(
                    not re.fullmatch(r"[a-f0-9]{64}", signature[k]) for k in ("train", "validation")
                )
            ):
                raise ValueError("Training recipe changed")
            signatures.append({k: v for k, v in signature.items() if k != "lr"})
            train = [m for m in proof["metrics"] if m["kind"] == "train"]
            val = [m for m in proof["metrics"] if m["kind"] == "validation"]
            if (
                [m["step"] for m in train] != list(range(1, step + 1))
                or [(m["epoch"], m["step"]) for m in val]
                != [(e, (e + 1) * 90) for e in range(epoch)]
                or any(m["epoch"] != (m["step"] - 1) // 90 for m in train)
                or len(proof["metrics"]) != step + epoch
                or any(m["update_skipped"] or not nonnegative(m["grad_norm"]) for m in train)
                or any(not nonnegative(m["loss"]) for m in proof["metrics"])
            ):
                raise ValueError("Incomplete/nonfinite training history")
            timing = proof["timing"]
            if any(not positive(timing[k]) for k in ("seconds", "peak_allocated_bytes")):
                raise ValueError("Invalid training timing")
            cp = proof["checkpoint"]
            if (
                cp["prefix"] != f"convergence-runs/{r['experiment']}/{arm}/{step}"
                or cp["readback_verified"] is not True
                or set(cp["files"])
                != {"last.pt", "config.json", "arguments.json", "metrics.jsonl", "timing.json"}
                or sum(s["bytes"] for s in cp["files"].values()) > 512 * 1024**2
                or cp["manifest_sha256"]
                != hashlib.sha256(encoded({"files": cp["files"]})).hexdigest()
            ):
                raise ValueError("Durable checkpoint export mismatch")
            for spec in cp["files"].values():
                if not positive(spec["bytes"]) or not re.fullmatch(r"[a-f0-9]{64}", spec["sha256"]):
                    raise ValueError("Invalid checkpoint descriptor")
            previous = cp["files"]["last.pt"]["sha256"]
        # The earlier history must be an exact prefix of the later checkpoints.
        for a, b in itertools.pairwise(proofs):
            if a["metrics"] != b["metrics"][: len(a["metrics"])]:
                raise ValueError("Resume altered metric history")
    if any(s != signatures[0] for s in signatures[1:]):
        raise ValueError("Unmatched data/configuration signatures")


def validate(result, sha, global_id, pod_id, experiment):
    if result.get("status") != "passed":
        raise ValueError("Worker did not report success")
    validate_report(result.get("report", {}), sha, global_id, pod_id, experiment)


def validate_report(r, sha, global_id, pod_id, experiment):
    """Audit retained data independently; never relabel a failed worker as passed."""
    storage = r.get("storage", {})
    if (
        r.get("kind") != "convergence-pilot"
        or r.get("commit") != sha
        or r.get("pod_id") != pod_id
        or r.get("experiment") != experiment
        or not re.fullmatch(r"\d+-\d+", experiment)
        or storage.get("global_volume_id") != global_id
        or storage.get("snapshot_sha256") != SNAPSHOT_SHA256
        or storage.get("verified_audio_files") != 500
        or r.get("dataset") != evidence()["resume"]["dataset"]
        or r.get("protocol") != PROTOCOL
        or r.get("initial_sha256") != evidence()["resume"]["initial_sha256"]
        or r.get("torch") != "2.8.0+cu128"
        or r.get("cuda") != "12.8"
        or r.get("museval") != "0.4.1"
        or any(r.get(k) is not True for k in ("complete", "matched_rng", "weights_unchanged"))
    ):
        raise ValueError("Training screen identity/protocol mismatch")
    validate_runtime(r["gpu_runtime"], MIG24)
    validate_training(r)
    if not positive(r["seconds"]) or len(r["budget_projections"]) != 1:
        raise ValueError("Missing runtime budget evidence")
    projection = r["budget_projections"][0]
    if (
        projection["accepted"] is not True
        or not positive(projection["projected_seconds"])
        or not projection["projected_seconds"] < projection["remaining_seconds"] <= 3600
    ):
        raise ValueError("Rejected measured budget")
    pinned = json.loads((ROOT / "docs/results/musdb18-training-provenance.json").read_text())
    expected = validation_inventory(r["provenance_json"], r["dataset"]["provenance_sha256"], pinned)
    coverage = r["coverage"]
    # Independently derive the actual one-crop/epoch offsets from the fixed manifest order.
    provenance = json.loads(r["provenance_json"])
    train_ids = pinned["split_track_ids"]["train"]
    tracks = {t["id"].removesuffix(".stem.mp4"): t for t in provenance["tracks"]}
    if coverage["epochs"] != 10 or coverage["crop_samples"] != 527360:
        raise ValueError("Crop geometry mismatch")
    if [t["id"] for t in coverage["tracks"]] != train_ids:
        raise ValueError("Training coverage selection/order mismatch")
    for index, track in enumerate(coverage["tracks"]):
        n, length = tracks[track["id"]]["samples"], coverage["crop_samples"]
        starts = [
            random.Random(f"17/{e}/{index}").randint(0, max(0, n - length)) for e in range(10)
        ]
        covered, end = 0, 0
        for start in sorted(starts):
            stop = min(n, start + length)
            covered += max(0, stop - max(start, end))
            end = max(end, stop)
        if track != {"id": track["id"], "samples": n, "starts": starts, "covered_samples": covered}:
            raise ValueError("Crop coverage differs from actual loader rule")
    labels = {VARIANTS[0]} | {f"{arm}_step{s}" for arm in ARMS for s in MILESTONES}
    if set(r["curves"]) != labels:
        raise ValueError("Missing quality curve milestone")
    for curve in r["curves"].values():
        if any(not positive(curve[k]) for k in ("seconds", "peak_allocated_bytes")):
            raise ValueError("Invalid quality timing")
        if [t["id"] for t in curve["tracks"]] != sorted(expected):
            raise ValueError("Incomplete quality curve songs")
        for t in curve["tracks"]:
            if (
                t["samples"] != expected[t["id"]]["samples"]
                or set(t["scores"]) != set(STEMS)
                or any(
                    isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                    for x in t["scores"].values()
                )
            ):
                raise ValueError("Invalid SI-SDR curve")
    if r["summary"] != summarize(r["records"], expected):
        raise ValueError("Summary mismatch")
    validate_baseline(r["summary"])
    for variant in VARIANTS:
        for record, track in zip(r["records"], r["curves"][variant]["tracks"], strict=True):
            for stem in STEMS:
                if (
                    record["scores"][variant][stem]["global_stereo_si_sdr_db"]
                    != track["scores"][stem]
                ):
                    raise ValueError("Final scores/curve disagree")
    names = {"report.json", "LISTENING.txt"} | {f"track-{i:02d}.json" for i in range(10)}
    if len(r["listening"]) != 2:
        raise ValueError("Missing listening samples")
    for i, sample in enumerate(r["listening"]):
        tid = sorted(expected)[i]
        clips = {f"sample-{i + 1}/reference-{s}.wav" for s in ("mixture", *STEMS)}
        clips |= {f"sample-{i + 1}/{v}-{s}.wav" for v in VARIANTS for s in STEMS}
        if (
            sample["id"] != tid
            or sample["samples"] != RATE * 8
            or sample["start_sample"] != (expected[tid]["samples"] - RATE * 8) // 2
            or not 0 < sample["gain"] <= 1
            or sample["files"] != sorted(clips)
        ):
            raise ValueError("Listening scope/shared gain mismatch")
        names |= clips
    bundle = r["bundle"]
    if (
        set(bundle["files"]) != names
        or not 0 < bundle["bytes"] <= 100 * 1024**2
        or not re.fullmatch(r"[a-f0-9]{64}", bundle["sha256"])
    ):
        raise ValueError("Invalid result bundle")
    for spec in bundle["files"].values():
        if not 0 < spec["bytes"] <= 100 * 1024**2 or not re.fullmatch(
            r"[a-f0-9]{64}", spec["sha256"]
        ):
            raise ValueError("Invalid result member")


def validate_baseline(summary):
    old = json.loads((ROOT / "docs/results/step90-validation-41196010747.json").read_text())
    for pair in summary.values():
        for metric, stems in pair["metrics"].items():
            for stem, values in stems.items():
                actual, previous = values[VARIANTS[0]], old["metrics"][metric][stem][VARIANTS[0]]
                for k in ("mean", "median"):
                    if not math.isclose(actual[k], previous[k], abs_tol=0.001, rel_tol=0):
                        raise ValueError("Official baseline failed reproduction gate")
                for k in ("total", "finite", "undefined", "positive_infinity", "negative_infinity"):
                    if actual[k] != previous[k]:
                        raise ValueError("Official baseline coverage changed")
