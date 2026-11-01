"""Predeclared two-model protocol and summaries, shared with the offline verifier."""

import math
import statistics as stats

from finetune_evaluation import PROTOCOL as PAIRED_PROTOCOL
from finetune_evaluation import RATE, STEMS, summarize_pair
from musdb_evaluation import decode_number, statistics

VARIANTS = ("official_converted", "htdemucs")
REPEATS = 3
PROTOCOL = {
    **PAIRED_PROTOCOL,
    "dataset": "MUSDB18 AAC; project 90/10 split; upstream training overlap unresolved",
    "seed": 17,
    "cpu_threads": 2,
    "repeats": REPEATS,
    "spleeter": "4 stems, 512 frames, 1024 bins, zero high-band extension",
    "htdemucs": {
        "signature": "955717e8",
        "shifts": 0,
        "split": True,
        "overlap": 0.25,
        "segment_seconds": 7.8,
        "workers": 0,
        "normalization": "upstream reversible mono-reference mean/std",
    },
    "warmup": "first sorted song, first 8 seconds, once per process",
    "processes": "one sequential independent process per model; CPU scoring afterward",
    "import_timing": "numeric/model dependency imports inside run(); excludes interpreter and shared protocol imports",
    "inference_timing": "host FP32 waveform through preprocessing, CUDA and CPU outputs; synchronized",
    "end_to_end_timing": "decode + inference + FLOAT WAV export; excludes import/load/warmup/scoring",
    "memory": "per-repeat reset CUDA allocated/reserved peaks including resident model; process peak RSS",
    "exports": "first repeat FLOAT WAV, exact 44.1k stereo samples; no clipping/rescaling",
    "baseline_tolerance_db": 0.001,
}


def summarize(records):
    summary = summarize_pair(records, variants=VARIANTS)
    # Always provide common finite-frame SDR, even when individual coverage is equal.
    common = {}
    for stem in STEMS:
        song_values = {v: [] for v in VARIANTS}
        counts = {}
        deltas = []
        for record in records:
            a, b = [
                [decode_number(x) for x in record["scores"][v][stem]["bsseval_v4"]["sdr"]]
                for v in VARIANTS
            ]
            pairs = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
            counts[record["id"]] = {"total": len(a), "common_finite": len(pairs)}
            medians = [stats.median(p[i] for p in pairs) if pairs else None for i in (0, 1)]
            for v, median in zip(VARIANTS, medians):
                song_values[v].append(median)
            deltas.append(medians[1] - medians[0] if pairs else None)
        common[stem] = {
            **{v: statistics(values) for v, values in song_values.items()},
            "paired_delta": statistics(deltas),
            "per_song_coverage": counts,
        }
    summary["common_finite_frame_sdr_db"] = common
    return summary


def timing_summary(measurements, rate):
    result = {}
    for model, report in measurements.items():
        duration = sum(t["samples"] for t in report["tracks"]) / RATE
        inference = sum(
            stats.median(r["inference_seconds"] for r in t["repeats"]) for t in report["tracks"]
        )
        end_to_end = sum(
            stats.median(r["end_to_end_seconds"] for r in t["repeats"]) for t in report["tracks"]
        )
        result[model] = {
            "audio_seconds": duration,
            "inference_seconds_sum_of_song_medians": inference,
            "end_to_end_seconds_sum_of_song_medians": end_to_end,
            "inference_rtf": inference / duration,
            "end_to_end_rtf": end_to_end / duration,
            "inference_audio_hours_per_wall_hour": duration / inference,
            "end_to_end_audio_hours_per_wall_hour": duration / end_to_end,
            "inference_usd_per_audio_hour": rate * inference / duration,
            "end_to_end_usd_per_audio_hour": rate * end_to_end / duration,
            "peak_allocated_bytes": max(
                r["peak_allocated_bytes"] for t in report["tracks"] for r in t["repeats"]
            ),
            "peak_reserved_bytes": max(
                r["peak_reserved_bytes"] for t in report["tracks"] for r in t["repeats"]
            ),
            "process_peak_rss_bytes": report["process_peak_rss_bytes"],
            "per_song": {
                t["id"]: {
                    k: {
                        "median": stats.median(r[k] for r in t["repeats"]),
                        "min": min(r[k] for r in t["repeats"]),
                        "max": max(r[k] for r in t["repeats"]),
                    }
                    for k in (
                        "inference_seconds",
                        "end_to_end_seconds",
                        "peak_allocated_bytes",
                        "peak_reserved_bytes",
                    )
                }
                for t in report["tracks"]
            },
        }
    return result


def validate_baseline(summary, previous):
    for metric in ("bsseval_v4_sdr_db", "global_stereo_si_sdr_db"):
        for stem in STEMS:
            a = summary["metrics"][metric][stem][VARIANTS[0]]
            b = previous["metrics"][metric][stem][VARIANTS[0]]
            for key in ("total", "finite", "undefined", "positive_infinity", "negative_infinity"):
                if a[key] != b[key]:
                    raise ValueError("Official baseline coverage changed")
            for key in ("mean", "median"):
                if not math.isclose(
                    decode_number(a[key]), decode_number(b[key]), abs_tol=0.001, rel_tol=0
                ):
                    raise ValueError("Official baseline failed 0.001 dB reproduction tolerance")
