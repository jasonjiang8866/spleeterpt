"""Measurement integrity and normalization regressions without renting a GPU."""

import copy
import hashlib
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    for folder in ("scripts", "runpod"):
        monkeypatch.syspath_prepend(str(ROOT / folder))
    return tuple(
        importlib.import_module(x)
        for x in ("benchmark_metrics", "benchmark_result", "separator_benchmark")
    )


def test_demucs_reversible_normalization_and_source_order(modules, monkeypatch):
    metrics, _, inference = modules
    waveform = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
    mean, std = waveform.mean(0).mean(), waveform.mean(0).std()
    model = SimpleNamespace(sources=["drums", "bass", "other", "vocals"])

    def apply(actual_model, mixture, **kwargs):
        assert actual_model is model
        torch.testing.assert_close(mixture[0], (waveform - mean) / std)
        assert kwargs == {
            "shifts": 0,
            "split": True,
            "overlap": 0.25,
            "segment": 7.8,
            "device": "cpu",
            "num_workers": 0,
        }
        return torch.stack([mixture * i for i in (1, 2, 3, 4)], dim=1)

    monkeypatch.setitem(sys.modules, "demucs.apply", SimpleNamespace(apply_model=apply))
    outputs = inference.demucs_separate(model, waveform, device="cpu")
    assert list(outputs) == list(metrics.STEMS)
    for i, source in enumerate(model.sources, 1):
        torch.testing.assert_close(outputs[source], (waveform - mean) * i + mean)
    assert outputs["vocals"].abs().max() > 1  # no clipping or CLI peak rescaling
    with pytest.raises(ValueError, match="Degenerate"):
        inference.demucs_separate(model, torch.ones(2, 100), device="cpu")


def test_common_frame_summary_and_paired_song_delta(modules):
    metrics, _, _ = modules
    records = []
    for i, values in enumerate((([0, None, 100], [2, 8, None]), ([10, 12, 14], [7, 9, 11]))):
        records.append(
            {
                "id": str(i),
                "scores": {
                    v: {
                        s: {
                            "global_stereo_si_sdr_db": float(i + j),
                            "bsseval_v4": {m: values[j] for m in ("sdr", "isr", "sir", "sar")},
                        }
                        for s in metrics.STEMS
                    }
                    for j, v in enumerate(metrics.VARIANTS)
                },
            }
        )
    summary = metrics.summarize(records)
    common = summary["common_finite_frame_sdr_db"]["vocals"]
    assert common["paired_delta"]["median"] == -0.5
    assert common["per_song_coverage"]["0"] == {"total": 3, "common_finite": 1}
    # Standard summary retains each model's own coverage; does not silently use common frames.
    assert summary["metrics"]["bsseval_v4_sdr_db"]["vocals"]["paired_delta"]["median"] == -24


def measurement(metrics, checker):
    repeat = {
        "decode_seconds": 1.0,
        "inference_seconds": 2.0,
        "export_seconds": 1.0,
        "end_to_end_seconds": 4.0,
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
    }
    expected = {"track": {"samples": 441000}}
    report = {
        "model": metrics.VARIANTS[0],
        "smoke": False,
        "state_unchanged": True,
        "protocol": metrics.PROTOCOL,
        "protocol_sha256": hashlib.sha256(checker.encoded(metrics.PROTOCOL)).hexdigest(),
        "weight_sha256": checker.weights()[metrics.VARIANTS[0]],
        "torch": "2.8.0+cu128",
        "cuda": "12.8",
        "gpu_runtime": {
            "requested_gpu": checker.MIG24,
            "name": checker.MIG24,
            "device_count": 1,
            "capability": [12, 0],
            "total_memory_bytes": 24 * 1024**3,
        },
        "import_seconds": 1.0,
        "load_seconds": 1.0,
        "warmup_seconds": 1.0,
        "seconds": 20.0,
        "process_peak_rss_bytes": 1000,
        "tracks": [
            {
                "id": "track",
                "samples": 441000,
                "repeats": [copy.deepcopy(repeat) for _ in range(3)],
                "files": {s: {"bytes": 100, "sha256": "a" * 64} for s in metrics.STEMS},
            }
        ],
    }
    return report, expected


def test_timing_rejection_and_cost_units(modules):
    metrics, checker, _ = modules
    report, expected = measurement(metrics, checker)
    checker.validate_measurement(report, metrics.VARIANTS[0], expected)
    summary = metrics.timing_summary({metrics.VARIANTS[0]: report}, 0.69)[metrics.VARIANTS[0]]
    assert summary["inference_rtf"] == 0.2
    assert summary["inference_usd_per_audio_hour"] == pytest.approx(0.138)
    mutations = [
        lambda r: r["tracks"][0]["repeats"].pop(),
        lambda r: r["tracks"][0].update(samples=440999),
        lambda r: r["tracks"][0]["repeats"][0].update(inference_seconds=float("nan")),
        lambda r: r["tracks"][0]["repeats"][0].update(end_to_end_seconds=1),
        lambda r: r["tracks"][0]["repeats"][0].update(peak_reserved_bytes=99),
        lambda r: r.update(weight_sha256="b" * 64),
        lambda r: r.update(state_unchanged=False),
        lambda r: r["gpu_runtime"].update(total_memory_bytes=96 * 1024**3),
    ]
    for mutate in mutations:
        bad = copy.deepcopy(report)
        mutate(bad)
        with pytest.raises(ValueError):
            checker.validate_measurement(bad, metrics.VARIANTS[0], expected)


def test_actual_museval_pair_with_benchmark_variants(modules):
    metrics, _, _ = modules
    score = importlib.import_module("musdb_evaluation").score_variants
    rng = np.random.default_rng(17)
    targets = {s: rng.normal(0, 0.03, (2, 44100)).astype(np.float32) for s in metrics.STEMS}
    targets["mixture"] = sum(targets.values())
    clean = {s: targets[s] + rng.normal(0, 0.001, targets[s].shape) for s in metrics.STEMS}
    record = {
        "id": "one",
        **score(
            targets,
            dict.fromkeys(metrics.VARIANTS, clean),
            metrics.STEMS,
            length_policy="strict",
            variant_names=metrics.VARIANTS,
        ),
    }
    summary = metrics.summarize([record])
    assert all(
        x["paired_delta"]["median"] == 0 for x in summary["metrics"]["bsseval_v4_sdr_db"].values()
    )


@pytest.mark.parametrize("missing", ["ffmpeg", "ffprobe"])
def test_missing_scoring_binary_rejected_before_import(modules, monkeypatch, missing):
    deps = importlib.import_module("benchmark_dependencies")
    monkeypatch.setattr(
        deps.shutil, "which", lambda name: None if name == missing else "/bin/" + name
    )
    imports = []
    monkeypatch.setattr(deps.importlib, "import_module", lambda name: imports.append(name))
    with pytest.raises(RuntimeError, match=missing):
        deps.check_scoring_runtime()
    assert imports == []


def test_scoring_runtime_real_import(modules):
    deps = importlib.import_module("benchmark_dependencies")
    deps.check_scoring_runtime()


def test_partial_checkpoint_cannot_be_accepted_as_benchmark(modules):
    metrics, checker, _ = modules
    report = {
        "kind": "benchmark-partial",
        "protocol_sha256": hashlib.sha256(checker.encoded(metrics.PROTOCOL)).hexdigest(),
        "phase": "smoke-htdemucs",
    }
    with pytest.raises(ValueError, match="identity"):
        checker.validate({"status": "passed", "report": report}, "a" * 40, "global1", "pod1", "1-1")
