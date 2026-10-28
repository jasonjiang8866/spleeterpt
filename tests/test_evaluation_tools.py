"""Analytic checks for evaluation metrics; no audio/model downloads required."""

import importlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")


@pytest.fixture
def evaluation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("evaluate_preview")


def test_si_sdr_known_orthogonal_noise(evaluation):
    reference = np.array([-1.0, 1.0, -1.0, 1.0])
    noise = np.array([-1.0, -1.0, 1.0, 1.0])
    score = evaluation.si_sdr(reference, reference + 0.5 * noise)
    assert score == pytest.approx(10 * np.log10(4))


def test_si_sdr_scale_and_dc_invariance(evaluation):
    reference = np.array([-1.0, 1.0, -1.0, 1.0])
    estimate = reference + np.array([-0.5, -0.5, 0.5, 0.5])
    assert evaluation.si_sdr(reference, 3 * estimate + 10) == pytest.approx(
        evaluation.si_sdr(reference, estimate)
    )
    assert evaluation.si_sdr(np.zeros(4), estimate) is None


def test_agreement_preserves_gain_error_and_reports_length(evaluation):
    reference = np.array([[1.0, -1.0, 1.0, -1.0]])
    candidate = 0.5 * reference[:, :3]
    report = evaluation.agreement(reference, candidate)
    assert report["reference_samples"] == 4
    assert report["candidate_samples"] == 3
    assert report["rmse"] == 0.5
    assert report["reference_snr_db"] == pytest.approx(10 * np.log10(4))
