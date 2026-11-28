import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def metric(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("parity_metrics").agreement


def test_strict_failures_are_not_hidden_by_si_sdr(metric):
    ref = np.array([[1.0, -1.0, 1.0, -1.0]])
    result = metric(ref, ref * 2)
    assert result["strict_failed_samples"] == 4
    assert not result["allclose_atol_1e-5_rtol_1e-4"]
    assert result["si_sdr_status"] == "exact_scaled_reference"
    assert result["rmse"] == 1


def test_silence_is_explicit_and_absolute_gate_still_applies(metric):
    ref = np.zeros((2, 8))
    result = metric(ref, ref + 2e-5)
    assert result["reference_si_sdr_db"] is None
    assert result["si_sdr_status"] == "undefined_silence_or_constant"
    assert result["near_silent_reference"]
    assert result["strict_failed_samples"] == 16


def test_si_sdr_and_distribution(metric):
    ref = np.array([[-1.0, 1.0, -1.0, 1.0]])
    got = ref + np.array([[-0.5, -0.5, 0.5, 0.5]])
    result = metric(ref, got)
    assert result["reference_si_sdr_db"] == pytest.approx(10 * np.log10(4))
    assert result["absolute_error_quantiles"]["p99"] == 0.5
    assert result["strict_failure_fraction"] == 1


def test_length_mismatch_cannot_pass(metric):
    result = metric(np.ones((2, 8)), np.ones((2, 7)))
    assert result["strict_failed_samples"] == 0
    assert not result["allclose_atol_1e-5_rtol_1e-4"]


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_rejected(metric, bad):
    with pytest.raises(ValueError, match="finite"):
        metric(np.ones((2, 8)), np.full((2, 8), bad))


def test_multiwindow_locations_distinguish_channel_window_and_source_repeat(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    parity = importlib.import_module("parity_metrics")
    config = SimpleNamespace(n_fft=8, hop_length=2, segment_frames=8, sample_rate=100)
    reference = np.zeros((2, 27))
    full = reference.copy()
    bounded = reference.copy()
    full[0, [7, 8, 13, 14, 24, 26]] = 2 * parity.ATOL
    bounded[0, [6, 9, 12, 14, 24, 25]] = -2 * parity.ATOL
    full[1, 13] = bounded[1, 13] = 3 * parity.ATOL
    full[1, 0] = parity.ATOL  # Equality to the tolerance passes.
    before = parity.agreement(reference, full)
    metrics, left = parity.agreement_with_failures(reference, full)
    _, right = parity.agreement_with_failures(reference, bounded)
    assert metrics == before and metrics["strict_failed_samples"] == 7

    report = parity.multiwindow_failure_locations({"drums": left}, {"drums": right}, config, 27, 11)
    assert report["sample_rate"] == 100 and report["source_repeat_starts"] == [11, 22]
    assert report["output_windows"] == [
        {"start": 0, "stop": 8, "overlap_stop": 0},
        {"start": 8, "stop": 24, "overlap_stop": 14},
        {"start": 24, "stop": 27, "overlap_stop": 27},
    ]
    stem = report["stems"]["drums"]
    assert stem["full"][0] == {
        "sample_indices": [7, 8, 13, 14, 24, 26],
        "omitted": 0,
        "by_window": [1, 3, 2],
        "in_overlap": [0, 2, 2],
    }
    assert stem["full"][1]["sample_indices"] == [13]
    assert stem["both_modes_by_channel"] == [2, 1]
    assert stem["bounded"][0]["sample_indices"] == [6, 9, 12, 14, 24, 25]


def test_dense_locations_keep_exact_counts_and_cross_mode_overlap(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    parity = importlib.import_module("parity_metrics")
    config = SimpleNamespace(n_fft=4096, hop_length=1024, segment_frames=512, sample_rate=44100)
    full = np.ones((2, 1300), dtype=bool)
    bounded = np.zeros_like(full)
    bounded[0, 700:] = True
    report = parity.multiwindow_failure_locations(
        {"bass": full}, {"bass": bounded}, config, 1300, 1300
    )
    stem = report["stems"]["bass"]
    assert report["source_repeat_starts"] == []
    assert stem["full"][0]["sample_indices"] == list(range(report["position_limit_per_channel"]))
    assert stem["full"][0]["omitted"] == 788 and stem["full"][0]["by_window"] == [1300]
    assert stem["both_modes_by_channel"] == [600, 0]
    assert stem["bounded"][1]["sample_indices"] == []
