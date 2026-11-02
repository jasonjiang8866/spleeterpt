import importlib
from pathlib import Path

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
