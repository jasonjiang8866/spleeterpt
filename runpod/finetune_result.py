"""Independently verify full-song selection, metrics, pinned weights and audio export."""

import hashlib
import json
import math
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BUNDLE = 100 * 1024**2


def load_evidence():
    return json.loads((ROOT / "docs/results/persistent-pilot-41189465452.json").read_text())


def validate_finetune(result, sha, volume_id, pod_id, *, ablation=False):
    sys.path.insert(0, str(ROOT / "scripts"))
    from finetune_evaluation import (
        PROTOCOL,
        RATE,
        STEMS,
        VARIANTS,
        summarize_pair,
        validate_records,
        validation_inventory,
    )

    if ablation:
        from bn_ablation import PROTOCOL, VARIANTS, summarize_pair, validate_records

    r = result.get("report", {})
    evidence = load_evidence()
    pinned = json.loads((ROOT / "docs/results/musdb18-training-provenance.json").read_text())
    if (
        result.get("status") != "passed"
        or r.get("kind") != ("bn-ablation" if ablation else "finetune-validation")
        or r.get("commit") != sha
        or r.get("volume_id") != volume_id
        or volume_id != evidence["volume"]["id"]
        or r.get("pod_id") != pod_id
        or r.get("experiment") != "41189465452-1"
        or r.get("results_directory")
        != f"/data/evaluations/41189465452-1/{'bn-' if ablation else ''}{pod_id}"
        or r.get("protocol") != PROTOCOL
        or r.get("dataset") != evidence["resume"]["dataset"]
        or r.get("backup") != evidence["resume"]["backup"]
        or r.get("initial_sha256") != evidence["resume"]["initial_sha256"]
        or r.get("weights_unchanged") is not True
        or r.get("torch") != "2.8.0+cu128"
        or r.get("cuda") != "12.8"
        or r.get("gpu") != "NVIDIA GeForce RTX 4090"
        or r.get("packages", {}).get("museval") != "0.4.1"
    ):
        raise RuntimeError("Validation identity, protocol, weights or runtime mismatch")
    expected = validation_inventory(r["provenance_json"], r["dataset"]["provenance_sha256"], pinned)
    validate_records(r["records"], expected)
    if r["summary"] != summarize_pair(r["records"]):
        raise RuntimeError("Validation summary differs from recomputed paired metrics")
    for k in ("seconds", "peak_allocated_bytes"):
        if isinstance(r[k], bool) or not math.isfinite(r[k]) or r[k] <= 0:
            raise RuntimeError("Invalid runtime measurement")
    expected_names = {"report.json", "LISTENING.txt"} | {f"track-{i:02d}.json" for i in range(10)}
    if ablation:
        from bn_ablation import validate_baselines, validate_proof

        validate_proof(r["bn_proof"])
        validate_baselines(
            r["summary"],
            json.loads((ROOT / "docs/results/step90-validation-41196010747.json").read_text()),
        )
    if len(r["listening"]) != (0 if ablation else 2):
        raise RuntimeError("Missing listening samples")
    for index, sample in enumerate(r["listening"]):
        track_id = sorted(expected)[index]
        names = {f"sample-{index + 1}/reference-{s}.wav" for s in ("mixture", *STEMS)}
        names |= {f"sample-{index + 1}/{v}-{s}.wav" for v in VARIANTS for s in STEMS}
        if (
            sample["id"] != track_id
            or sample["samples"] != RATE * 8
            or sample["start_sample"] != (expected[track_id]["samples"] - RATE * 8) // 2
            or not 0 < sample["gain"] <= 1
            or sample["files"] != sorted(names)
        ):
            raise RuntimeError("Listening selection, interval or shared gain mismatch")
        expected_names |= names
    b = r["bundle"]
    if (
        set(b["files"]) != expected_names
        or not 0 < b["bytes"] <= MAX_BUNDLE
        or not re.fullmatch(r"[a-f0-9]{64}", b["sha256"])
    ):
        raise RuntimeError("Unexpected evaluation bundle")
    for spec in b["files"].values():
        if not 0 < spec["bytes"] <= MAX_BUNDLE or not re.fullmatch(r"[a-f0-9]{64}", spec["sha256"]):
            raise RuntimeError("Invalid bundle member descriptor")


def validate_export(path, report):
    from quality import digest

    b = report["bundle"]
    if path.stat().st_size != b["bytes"] or digest(path) != b["sha256"]:
        raise RuntimeError("Evaluation export checksum mismatch")
    with tarfile.open(path, "r:") as tar:
        members = tar.getmembers()
        if len(members) != len(b["files"]) or {m.name for m in members} != set(b["files"]):
            raise RuntimeError("Export inventory mismatch")
        for m in members:
            spec = b["files"][m.name]
            if not m.isfile() or m.size != spec["bytes"]:
                raise RuntimeError("Invalid export member")
            data = tar.extractfile(m).read()
            if hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise RuntimeError("Export member checksum mismatch")
            if m.name == "report.json" and json.loads(data) != {
                k: v for k, v in report.items() if k != "bundle"
            }:
                raise RuntimeError("Exported report mismatch")
            if (
                m.name.startswith("track-")
                and json.loads(data) != report["records"][int(m.name[6:8])]
            ):
                raise RuntimeError("Exported track mismatch")


def download_export(url, token, report, path):
    from transfer import read_chunk

    b = report["bundle"]
    if not 0 < b["bytes"] <= MAX_BUNDLE:
        raise RuntimeError("Export too large")
    temporary = path.with_suffix(".part")
    try:
        with temporary.open("wb") as stream:
            for offset in range(0, b["bytes"], 4 * 1024**2):
                stream.write(read_chunk(url, token, offset, min(4 * 1024**2, b["bytes"] - offset)))
        validate_export(temporary, report)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
