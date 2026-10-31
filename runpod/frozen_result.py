"""Runner-side acceptance for the matched training experiment."""

import hashlib
import json
import math
import re
import sys
from pathlib import Path

from global_inputs import encoded, evidence
from gpu_profile import MIG24, validate_runtime

ROOT = Path(__file__).resolve().parents[1]


def validate_frozen(result, sha, pod_id, storage, experiment):
    sys.path.insert(0, str(ROOT / "scripts"))
    from finetune_evaluation import RATE, STEMS, validation_inventory
    from frozen_bn_evaluation import PROTOCOL, VARIANTS, summarize

    r = result.get("report", {})
    if (
        result.get("status") != "passed"
        or r.get("kind") != "frozen-bn-pilot"
        or r.get("commit") != sha
        or r.get("pod_id") != pod_id
        or r.get("experiment") != experiment
        or not re.fullmatch(r"\d+-\d+", experiment)
        or r.get("storage") != storage
        or storage.get("verified_audio_files") != 500
        or r.get("dataset") != evidence()["resume"]["dataset"]
        or r.get("protocol") != PROTOCOL
        or r.get("initial_sha256") != evidence()["resume"]["initial_sha256"]
        or r.get("torch") != "2.8.0+cu128"
        or r.get("cuda") != "12.8"
        or r.get("museval") != "0.4.1"
        or r.get("matched_rng") is not True
        or r.get("weights_unchanged") is not True
    ):
        raise ValueError("Matched training identity/protocol mismatch")
    validate_runtime(r["gpu_runtime"], MIG24)
    if set(r["arms"]) != {"normal", "frozen"}:
        raise ValueError("Missing matched training arm")
    for arm, proof in r["arms"].items():
        if (
            proof["cursor"] != [90, 1, 0]
            or proof["optimizer_steps"] != [90]
            or proof["finite"] is not True
            or proof["bn_affine_changed"] is not True
            or proof["parameters_changed"] is not True
            or proof["bn_buffers_unchanged"] != (arm == "frozen")
            or (proof["bn_fingerprint"] == r["initial_bn_fingerprint"]) != (arm == "frozen")
        ):
            raise ValueError("Training state acceptance failed")
        signature = proof["signature"]
        expected_recipe = {
            k: v
            for k, v in PROTOCOL["recipe"].items()
            if k
            in (
                "batch_size",
                "crops_per_track",
                "validation_crops",
                "lr",
                "seed",
                "grad_clip",
                "deterministic",
            )
        }
        if any(signature[k] != v for k, v in expected_recipe.items()) or (
            signature["precision"] != "fp32" or signature["freeze_bn_stats"] != (arm == "frozen")
        ):
            raise ValueError("Training recipe changed")
        train = [m for m in proof["metrics"] if m["kind"] == "train"]
        val = [m for m in proof["metrics"] if m["kind"] == "validation"]
        if (
            [m["step"] for m in train] != list(range(1, 91))
            or len(val) != 1
            or val[0]["step"] != 90
            or any(m["update_skipped"] for m in train)
            or any(not math.isfinite(m["loss"]) for m in proof["metrics"])
        ):
            raise ValueError("Training steps or losses are incomplete")
        cp = proof["checkpoint"]
        if (
            cp["prefix"] != f"frozen-bn-runs/{experiment}/{arm}"
            or cp["readback_verified"] is not True
            or set(cp["files"])
            != {"last.pt", "arguments.json", "config.json", "metrics.jsonl", "timing.json"}
            or sum(s["bytes"] for s in cp["files"].values()) > 512 * 1024**2
            or cp["manifest_sha256"] != hashlib.sha256(encoded({"files": cp["files"]})).hexdigest()
        ):
            raise ValueError("Durable checkpoint export mismatch")
        for spec in cp["files"].values():
            if not 0 < spec["bytes"] or not re.fullmatch(r"[a-f0-9]{64}", spec["sha256"]):
                raise ValueError("Invalid checkpoint file descriptor")
    signatures = [
        {k: v for k, v in r["arms"][arm]["signature"].items() if k != "freeze_bn_stats"}
        for arm in ("normal", "frozen")
    ]
    if signatures[0] != signatures[1]:
        raise ValueError("Unmatched data or hyperparameters")
    pinned = json.loads((ROOT / "docs/results/musdb18-training-provenance.json").read_text())
    expected = validation_inventory(r["provenance_json"], r["dataset"]["provenance_sha256"], pinned)
    if r["summary"] != summarize(r["records"], expected):
        raise ValueError("Summary differs from independently recomputed metrics")
    old = json.loads((ROOT / "docs/results/step90-validation-41196010747.json").read_text())
    for metric, stems in r["summary"]["frozen_vs_official"]["metrics"].items():
        for stem, values in stems.items():
            actual, previous = (
                values["official_converted"],
                old["metrics"][metric][stem]["official_converted"],
            )
            for key in ("mean", "median"):
                if not math.isclose(actual[key], previous[key], abs_tol=0.001, rel_tol=0):
                    raise ValueError("Official baseline failed predeclared reproduction gate")
            for key in ("total", "finite", "undefined", "positive_infinity", "negative_infinity"):
                if actual[key] != previous[key]:
                    raise ValueError("Official baseline coverage changed")
    for pair in r["summary"].values():
        for values in pair["metrics"]["global_stereo_si_sdr_db"].values():
            if any(values[v]["finite"] != 10 for v in ("official_converted", "finetuned_step90")):
                raise ValueError("Nonfinite SI-SDR")
    for key in ("seconds", "evaluation_peak_allocated_bytes"):
        if isinstance(r[key], bool) or not math.isfinite(r[key]) or r[key] <= 0:
            raise ValueError("Invalid runtime measurement")
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
            raise ValueError("Listening scope or shared gain mismatch")
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
