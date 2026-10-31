"""Paired validation protocol; independent of the four-way port-parity experiment."""

import hashlib
import json
import math

from musdb_evaluation import BSS_METRICS, decode_number, encode_number, statistics, track_score

VARIANTS = ("official_converted", "finetuned_step90")
STEMS = ("vocals", "drums", "bass", "other")
RATE = 44100
PROTOCOL = {
    "scope": "all 10 project validation songs, complete recordings",
    "sample_rate": RATE,
    "start_sample": 0,
    "length_policy": "strict",
    "precision": "fp32",
    "tf32": False,
    "bsseval": "museval 0.4.1 v4 source images, fixed 512-tap filters, no permutation",
    "window_samples": RATE,
    "hop_samples": RATE,
    "padding": False,
    "aggregation": "median 1-second frames per song; median songs per stem",
    "paired_delta": "candidate minus baseline per-song median; median paired differences",
    "si_sdr": "global stereo, mean removed, one scalar; separate from BSSEval SDR",
    "undefined": "null; infinities as strings; counts explicit; paired deltas finite-only",
    "listening": "first two sorted validation IDs, centered 8 seconds, shared gain per song",
    "test_pool_used": False,
}


def validation_inventory(provenance_json, expected_sha256, pinned):
    """Use the exact decoded files used in training, not another machine's WAV headers."""
    if hashlib.sha256(provenance_json.encode()).hexdigest() != expected_sha256:
        raise ValueError("Training dataset provenance hash mismatch")
    provenance = json.loads(provenance_json)
    source = {t["id"]: (t["source_sha256"], t["samples"]) for t in pinned["tracks"]}
    if (
        len(provenance["tracks"]) != 100
        or {t["id"]: (t["source_sha256"], t["samples"]) for t in provenance["tracks"]} != source
        or provenance["archive_sha256"] != pinned["archive_sha256"]
        or provenance["test_audio_extracted"] is not False
    ):
        raise ValueError("Persistent source inventory differs from official training pool")
    return {
        t["id"].removesuffix(".stem.mp4"): t
        for t in provenance["tracks"]
        if t["id"].removesuffix(".stem.mp4") in pinned["split_track_ids"]["validation"]
    }


def summarize_pair(records):
    result = {"track_count": len(records), "metrics": {}, "frame_coverage": {}}
    for metric in ("bsseval_v4_sdr_db", "global_stereo_si_sdr_db"):
        result["metrics"][metric] = {}
        for stem in STEMS:
            values = {v: [track_score(r, v, stem, metric) for r in records] for v in VARIANTS}
            deltas = []
            for a, b in zip(values[VARIANTS[0]], values[VARIANTS[1]]):
                a, b = decode_number(a), decode_number(b)
                deltas.append(b - a if math.isfinite(a) and math.isfinite(b) else None)
            finite = [d for d in deltas if d is not None]
            result["metrics"][metric][stem] = {
                **{v: statistics(x) for v, x in values.items()},
                "paired_delta": {
                    **statistics(deltas),
                    "better": sum(d > 0 for d in finite),
                    "worse": sum(d < 0 for d in finite),
                    "equal": sum(d == 0 for d in finite),
                    "per_track": dict(zip([r["id"] for r in records], deltas)),
                },
            }
    for v in VARIANTS:
        result["frame_coverage"][v] = {
            stem: {
                m: statistics(
                    [x for r in records for x in r["scores"][v][stem]["bsseval_v4"][m]],
                    counts_only=True,
                )
                for m in BSS_METRICS
            }
            for stem in STEMS
        }
    return result


def validate_records(records, expected):
    """Reject selective songs, cropped audio, missing stems/windows and invalid scores."""
    if [r["id"] for r in records] != sorted(expected):
        raise ValueError("Validation selection differs from pinned ten-song split")
    for r in records:
        spec = expected[r["id"]]
        n = spec["samples"]
        if (
            r["input_sha256"] != {s: x["sha256"] for s, x in spec["files"].items()}
            or r["decoded_samples"] != n
            or r["evaluated_samples"] != n
            or r["segment_start_sample"] != 0
            or r["discarded_reference_tail_samples"] != 0
            or r["output_samples"] != {v: dict.fromkeys(STEMS, n) for v in VARIANTS}
            or set(r["scores"]) != set(VARIANTS)
        ):
            raise ValueError("Input identity, length or variant mismatch")
        for v in VARIANTS:
            if set(r["scores"][v]) != set(STEMS):
                raise ValueError("Missing stem")
            for stem in STEMS:
                scores = r["scores"][v][stem]
                decode_number(scores["global_stereo_si_sdr_db"])
                if set(scores["bsseval_v4"]) != set(BSS_METRICS):
                    raise ValueError("Missing metric")
                for series in scores["bsseval_v4"].values():
                    if len(series) != max(1, n // RATE):
                        raise ValueError("Missing one-second metric windows")
                    for x in series:
                        if encode_number(decode_number(x)) != x:
                            raise ValueError("Invalid metric encoding")
