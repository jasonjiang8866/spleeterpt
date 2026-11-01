import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("benchmark_bounded")


def test_silence_exact_and_shape_validation(metrics, tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(a, np.zeros((100, 2)), 44100, subtype="FLOAT")
    sf.write(b, np.zeros((100, 2)), 44100, subtype="FLOAT")
    m = metrics.paired_metrics(a, b)
    assert m["reference_rms"] == 0 and m["legacy_reference_si_sdr_db"] is None
    assert metrics.acceptance(m, SimpleNamespace(max_abs_error=2e-6, min_si_sdr=100)) == "exact"
    sf.write(b, np.zeros((99, 2)), 44100, subtype="FLOAT")
    with pytest.raises(ValueError, match="shape"):
        metrics.paired_metrics(a, b)


def test_quiet_stems_are_not_false_failures_or_blank_checks(metrics, tmp_path):
    rng = np.random.default_rng(19)
    x = rng.normal(0, 3e-6, (10000, 2))
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(a, x, 44100, subtype="FLOAT")
    sf.write(b, x + rng.normal(0, 6e-11, x.shape), 44100, subtype="FLOAT")
    m = metrics.paired_metrics(a, b)
    args = SimpleNamespace(max_abs_error=2e-6, min_si_sdr=100)
    assert 90 < m["legacy_reference_si_sdr_db"] < 100
    assert metrics.acceptance(m, args) == "near_silent_absolute"
    sf.write(b, np.zeros_like(x), 44100, subtype="FLOAT")
    assert metrics.acceptance(metrics.paired_metrics(a, b), args).startswith("failed")
    m.update(reference_rms=0.01)
    assert metrics.acceptance(m, args) == "failed_si_sdr"
