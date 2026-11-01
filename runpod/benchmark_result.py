"""Independent identity, selection, timing, metric and bundle validation on Actions."""

import hashlib
import json
import math
import re
import sys

from global_inputs import encoded, evidence, manifest
from gpu_profile import MIG24, validate_runtime
from quality import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_metrics import RATE, REPEATS, VARIANTS, validate_baseline
from finetune_evaluation import validate_records, validation_inventory
from probe_htdemucs import COMMIT
from probe_htdemucs import SHA256 as DEMUCS_SHA


def positive(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )


def profile(vocal=False):
    import importlib

    return importlib.import_module("vocal_metrics" if vocal else "benchmark_metrics")


def weights(vocal=False):
    if vocal:
        return profile(True).WEIGHTS
    return {VARIANTS[0]: evidence()["resume"]["initial_sha256"], VARIANTS[1]: DEMUCS_SHA}


def validate_measurement(r, variant, expected, *, smoke=False, vocal=False):
    p = profile(vocal)
    protocol, stems = p.PROTOCOL, p.STEMS
    if variant not in p.VARIANTS:
        raise ValueError("Unexpected model for benchmark profile")
    if vocal:
        from roformer_kim import recipe_sha256

        if (
            r.get("profile") != "vocal"
            or r.get("rss_source") != "/proc/self/status VmHWM"
            or r.get("roformer_recipe_sha256")
            != (recipe_sha256() if variant == "melband_kim" else None)
        ):
            raise ValueError("Vocal implementation/memory measurement identity mismatch")
    if r.get("state_unchanged") is not True:
        raise ValueError("Model state changed or was not checked")
    if r.get("demucs_source_commit") != (COMMIT if variant == "htdemucs" else None):
        raise ValueError("Model source commit mismatch")
    if (
        r.get("model") != variant
        or r.get("smoke") != smoke
        or r.get("protocol") != protocol
        or r.get("protocol_sha256") != hashlib.sha256(encoded(protocol)).hexdigest()
        or r.get("weight_sha256") != weights(vocal)[variant]
        or r.get("torch") != "2.8.0+cu128"
        or r.get("cuda") != "12.8"
    ):
        raise ValueError("Model timing identity mismatch")
    validate_runtime(r["gpu_runtime"], MIG24)
    for key in (
        "import_seconds",
        "load_seconds",
        "warmup_seconds",
        "seconds",
        "process_peak_rss_bytes",
    ):
        if not positive(r[key]):
            raise ValueError("Missing positive process measurement")
    ids = sorted(expected)[:1] if smoke else sorted(expected)
    if [t["id"] for t in r["tracks"]] != ids:
        raise ValueError("Incomplete timed track selection")
    for t in r["tracks"]:
        if (
            t["samples"] != expected[t["id"]]["samples"]
            or len(t["repeats"]) != (1 if smoke else REPEATS)
            or set(t["files"]) != set(stems)
        ):
            raise ValueError("Missing full-song timing repeats/outputs")
        for file in t["files"].values():
            if not positive(file["bytes"]) or not re.fullmatch(r"[a-f0-9]{64}", file["sha256"]):
                raise ValueError("Invalid output descriptor")
        for run in t["repeats"]:
            if set(run) != {
                "decode_seconds",
                "inference_seconds",
                "export_seconds",
                "end_to_end_seconds",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            } or not all(positive(x) for x in run.values()):
                raise ValueError("Invalid repeat measurement")
            summed = sum(run[k] for k in ("decode_seconds", "inference_seconds", "export_seconds"))
            if (
                not math.isclose(summed, run["end_to_end_seconds"], rel_tol=1e-9, abs_tol=1e-9)
                or not run["peak_allocated_bytes"]
                <= run["peak_reserved_bytes"]
                <= r["gpu_runtime"]["total_memory_bytes"]
            ):
                raise ValueError("Inconsistent timing or memory measurement")


def validate(result, sha, global_id, pod_id, experiment, *, vocal=False):
    p = profile(vocal)
    protocol, stems, variants = p.PROTOCOL, p.STEMS, p.VARIANTS
    r = result.get("report", {})
    if r.get("protocol_sha256") != hashlib.sha256(encoded(protocol)).hexdigest():
        raise ValueError("Protocol checksum mismatch")
    if (
        result.get("status") != "passed"
        or r.get("kind") != ("spleeter2-melband-kim" if vocal else "spleeter4-htdemucs")
        or r.get("commit") != sha
        or r.get("pod_id") != pod_id
        or r.get("experiment") != experiment
        or r.get("protocol") != protocol
        or r.get("weight_sha256") != weights(vocal)
    ):
        raise ValueError("Benchmark identity/protocol mismatch")
    full = manifest(r["provenance_json"])
    files = {k: v for k, v in full["files"].items() if k.startswith("musdb18/")}
    storage = r["storage"]
    if storage != {
        "mode": "global-staged",
        "global_volume_id": global_id,
        "snapshot_sha256": hashlib.sha256(encoded(full)).hexdigest(),
        "verified_audio_files": 50,
        "staged_files": 51,
        "staged_bytes": sum(s["bytes"] for s in files.values()),
        "stage_seconds": storage.get("stage_seconds"),
        "test_pool_used": False,
    } or not positive(storage["stage_seconds"]):
        raise ValueError("Global validation staging proof mismatch")
    pinned = json.loads((ROOT / "docs/results/musdb18-training-provenance.json").read_text())
    expected = validation_inventory(
        r["provenance_json"], evidence()["resume"]["dataset"]["provenance_sha256"], pinned
    )
    if set(r["measurements"]) != set(variants):
        raise ValueError("Missing model process")
    for variant in variants:
        validate_measurement(r["measurements"][variant], variant, expected, vocal=vocal)
    validate_records(r["records"], expected, variants=variants, stems=stems)
    if r["summary"] != p.summarize(r["records"]):
        raise ValueError("Summary differs from independently recomputed paired scores")
    if not vocal:
        validate_baseline(
            r["summary"],
            json.loads((ROOT / "docs/results/step90-validation-41196010747.json").read_text()),
        )
    else:
        for name, version in p.DEPENDENCIES.items():
            if r["packages"].get(name) != version:
                raise ValueError("Vocal dependency version mismatch: " + name)
    if r["packages"].get("museval") != "0.4.1" or not all(
        positive(r[k]) for k in ("worker_seconds", "scoring_seconds")
    ):
        raise ValueError("Scoring runtime mismatch")
    names = {"report.json", "LISTENING.txt", "ANSWERS.json"} | {
        f"track-{i:02d}.json" for i in range(10)
    }
    if len(r["listening"]) != 2:
        raise ValueError("Missing listening clips")
    for i, clip in enumerate(r["listening"]):
        track_id = sorted(expected)[i]
        clip_names = {
            f"sample-{i + 1}/{g}-{s}.wav" for g in ("A", "B", "reference") for s in stems
        } | {f"sample-{i + 1}/reference-mixture.wav"}
        if (
            clip["id"] != track_id
            or clip["samples"] != RATE * 8
            or clip["start_sample"] != (expected[track_id]["samples"] - RATE * 8) // 2
            or not positive(clip["gain"])
            or clip["gain"] > 1
            or clip["files"] != sorted(clip_names)
        ):
            raise ValueError("Listening selection/gain mismatch")
        names |= clip_names
    b = r["bundle"]
    if (
        set(b["files"]) != names
        or not positive(b["bytes"])
        or b["bytes"] > 100 * 1024**2
        or not re.fullmatch(r"[a-f0-9]{64}", b["sha256"])
    ):
        raise ValueError("Unexpected export bundle")
    for file in b["files"].values():
        if (
            not positive(file["bytes"])
            or file["bytes"] > 100 * 1024**2
            or not re.fullmatch(r"[a-f0-9]{64}", file["sha256"])
        ):
            raise ValueError("Invalid bundle member")
